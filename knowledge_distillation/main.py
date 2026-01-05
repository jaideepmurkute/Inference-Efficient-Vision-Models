import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_distillation.kd_config import KDConfig
from knowledge_distillation.utils import *
from knowledge_distillation.train import train_kd_one_epoch, validate, test

def main():
    cfg = KDConfig()
    
    # Setup Logger
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    
    # Set Seed
    set_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Build Data Paths
    data_paths = build_img_paths(cfg)
    
    # Define Teacher Experiment Path (Assumed relative location)
    # You might want to make this configurable
    teacher_exp_name = "exp_1" 
    teacher_output_root = os.path.join("..", "teacher_training", "output", teacher_exp_name)
    
    if cfg.choice == 1:
        img_paths = np.array(data_paths['train']['img_paths'])
        cls_ids = np.array(data_paths['train']['cls_ids'])
        
        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)
        
        for fold_id in range(cfg.num_folds):
            logger.info(f"Training Fold {fold_id}")
            
            train_loader = get_dataloader(cfg, split_type='train', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['train']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['train']])
            
            val_loader = get_dataloader(cfg, split_type='validation', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['validation']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['validation']])

            # --- TEACHER SETUP ---
            logger.info(f"Loading Teacher Model: {cfg.teacher_model} for Fold {fold_id}")
            teacher_model = timm.create_model(cfg.teacher_model, pretrained=True, num_classes=cfg.num_classes)
            teacher_model.to(device)
            teacher_model.eval()
            
            # Construct path to teacher checkpoint for this fold
            teacher_ckpt_path = os.path.join(teacher_output_root, f"fold_{fold_id}", "model_best.pth")
            
            if os.path.exists(teacher_ckpt_path):
                logger.info(f"Loading teacher weights from: {teacher_ckpt_path}")
                teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
            else:
                logger.error(f"Teacher checkpoint not found at {teacher_ckpt_path}. Cannot perform KD without teacher.")
                return

            # --- STUDENT SETUP ---
            logger.info(f"Creating Student Model: {cfg.student_model}")
            student_model = timm.create_model(cfg.student_model, pretrained=True, num_classes=cfg.num_classes)
            student_model.to(device)

            # Optimization
            criterion_ce = nn.CrossEntropyLoss()
            criterion_kd = nn.KLDivLoss(reduction='batchmean')
            optimizer = optim.AdamW(student_model.parameters(), lr=cfg.learning_rate)
            
            # Training Loop
            best_acc = 0.0
            training_log = {
                'train': {'loss': [], 'accuracy': []},
                'validation': {'loss': [], 'accuracy': []},
                'epoch_times': []
            }
            
            for epoch in range(cfg.epochs):
                train_loss, train_acc, epoch_time = train_kd_one_epoch(
                    student_model, teacher_model, train_loader, optimizer, 
                    criterion_ce, criterion_kd, cfg.alpha, cfg.temperature, 
                    device, logger
                )
                val_loss, val_acc = validate(student_model, val_loader, criterion_ce, device)
                
                logger.info(
                    f"Epoch [{epoch+1}/{cfg.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Update Log
                training_log['train']['loss'].append(train_loss)
                training_log['train']['accuracy'].append(train_acc)
                training_log['validation']['loss'].append(val_loss)
                training_log['validation']['accuracy'].append(val_acc)
                training_log['epoch_times'].append(epoch_time)
                
                # Save Best
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(cfg, model=student_model, fold_id=fold_id, suffix="best")
                    logger.info(f"Saved new best student model with Acc: {best_acc:.4f}")
                
                # Save Last
                save_checkpoint(cfg, model=student_model, fold_id=fold_id, suffix="last")

            logger.info(f"Fold {fold_id} Completed. Best Student Acc: {best_acc:.4f}")
            
            # Save Log
            save_checkpoint(cfg, fold_id=fold_id, training_log=training_log)
            visualize_training_log(cfg, training_log, fold_id=fold_id)
            print('-' * 50)

    elif cfg.choice == 2:
        # TESTING Loop
        logger.info("Starting Student Testing...")
        data_type = 'test'
        
        if data_type in ['train', 'validation']:
             img_paths = np.array(data_paths[data_type]['img_paths'])
             cls_ids = np.array(data_paths[data_type]['cls_ids'])
             
             fold_idx_dict = load_checkpoint(cfg, load_type='fold_idx_dict')
             
             for fold_id in fold_idx_dict.keys():
                 logger.info(f"Evaluating Fold {fold_id}")
                 
                 # Data
                 curr_img_paths = img_paths[fold_idx_dict[fold_id][data_type]]
                 curr_cls_ids = cls_ids[fold_idx_dict[fold_id][data_type]]
                 data_loader = get_dataloader(cfg, split_type=data_type, img_paths=curr_img_paths, cls_ids=curr_cls_ids)
                 
                 # Model
                 student_model = timm.create_model(cfg.student_model, pretrained=False, num_classes=cfg.num_classes)
                 student_model.to(device)
                 student_model = load_checkpoint(cfg, load_type='model', model=student_model, fold_id=fold_id, suffix=cfg.test_ckpt_type)
                 
                 test_acc = test(student_model, data_loader, device, logger)
                 logger.info(f"Fold {fold_id} {data_type} Accuracy: {test_acc:.4f}")

        elif data_type == 'test':
            # For pure test set, we could ensemble or pick one fold. For now, let's just test Fold 0 (or all folds and average)
            # The previous 'main.py' logic was simple single split. 
            # I'll loop through all available trained folds and evaluate them on the test set.
            
            img_paths = np.array(data_paths[data_type]['img_paths'])
            cls_ids = np.array(data_paths[data_type]['cls_ids'])
            data_loader = get_dataloader(cfg, split_type=data_type, img_paths=img_paths, cls_ids=cls_ids)
            
            for fold_id in range(cfg.num_folds):
                logger.info(f"Evaluating Fold {fold_id} Student on Test Set")
                
                try:
                    student_model = timm.create_model(cfg.student_model, pretrained=False, num_classes=cfg.num_classes)
                    student_model.to(device)
                    student_model = load_checkpoint(cfg, load_type='model', model=student_model, fold_id=fold_id, suffix=cfg.test_ckpt_type)
                    
                    test_acc = test(student_model, data_loader, device, logger)
                    logger.info(f"Fold {fold_id} Test Accuracy: {test_acc:.4f}")
                except Exception as e:
                    logger.warning(f"Skipping Fold {fold_id}: {e}")

if __name__ == "__main__":
    main()
