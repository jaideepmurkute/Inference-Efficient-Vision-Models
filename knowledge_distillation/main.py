import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_distillation.kd_config import KDConfig
from knowledge_distillation.utils import *
from knowledge_distillation.train import train_kd_one_epoch, validate, test

def create_model_safe(model_name, use_timm, pretrained, num_classes, logger):
    logger.info(f"Creating model {model_name} (use_timm={use_timm})...")
    if use_timm:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        if not hasattr(torchvision.models, model_name):
             raise ValueError(f"Torchvision model {model_name} not found")
        
        weights = "DEFAULT" if pretrained else None
        model_fn = getattr(torchvision.models, model_name)
        model = model_fn(weights=weights)
        
        # Replace Head
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        else:
             logger.warning(f"Could not automatically find classifier head for {model_name}.")
    return model

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
    teacher_output_root = os.path.join("..", "teacher_training", "output", cfg.teacher_exp_name)
    
    if cfg.choice == 1:
        img_paths = np.array(data_paths['train']['img_paths'])
        cls_ids = np.array(data_paths['train']['cls_ids'])
        
        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)
        
        for fold_id in range(cfg.num_folds):
            logger.info(f"--- Training Fold {fold_id} ---")
            
            train_loader = get_dataloader(cfg, split_type='train', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['train']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['train']])
            
            val_loader = get_dataloader(cfg, split_type='validation', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['validation']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['validation']])

            # --- TEACHER SETUP ---
            logger.info(f"Loading Teacher Model: {cfg.teacher_model} for Fold {fold_id}")
            # Ensure teacher uses the same 'use_timm' setting? 
            # Usually teacher (ResNet50) and student (ResNet18) are both Torchvision now.
            use_timm_flag = getattr(cfg, 'use_timm', True) # Default to True if not set, but we set to False in config.
            
            teacher_model = create_model_safe(cfg.teacher_model, use_timm_flag, pretrained=False, num_classes=cfg.num_classes, logger=logger)
            teacher_model.to(device)
            teacher_model.eval()
            
            # Construct path to teacher checkpoint for this fold
            teacher_ckpt_path = os.path.join(teacher_output_root, f"fold_{fold_id}", "model_best.pth")
            
            if os.path.exists(teacher_ckpt_path):
                logger.info(f"Loading teacher weights from: {teacher_ckpt_path}")
                state = torch.load(teacher_ckpt_path, map_location=device)
                
                # Check for DataParallel 'module.' prefix just in case
                new_state = {}
                for k, v in state.items():
                    if k.startswith('module.'): new_state[k[7:]] = v
                    else: new_state[k] = v
                
                teacher_model.load_state_dict(new_state)
            else:
                logger.error(f"Teacher checkpoint not found at {teacher_ckpt_path}. Cannot perform KD without teacher.")
                return

            # --- STUDENT SETUP ---
            logger.info(f"Creating Student Model: {cfg.student_model}")
            student_model = create_model_safe(cfg.student_model, use_timm_flag, pretrained=True, num_classes=cfg.num_classes, logger=logger)
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
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
                )
                
                training_log['train']['loss'].append(train_loss)
                training_log['train']['accuracy'].append(train_acc)
                training_log['validation']['loss'].append(val_loss)
                training_log['validation']['accuracy'].append(val_acc)
                training_log['epoch_times'].append(epoch_time)
                
                # Save Best Student
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(cfg, model=student_model, fold_id=fold_id, suffix="best", training_log=training_log)
                    logger.info(f"Saved Best Student Model (Acc: {best_acc:.2f}%)")

            # Save Last
            save_checkpoint(cfg, model=student_model, fold_id=fold_id, suffix="last", training_log=training_log)
            
            # Visualize (Optional if utils allows)
            visualize_training_log(cfg, training_log, fold_id=fold_id)

    elif cfg.choice == 2:
        logger.info("Running Test on all folds...")
        # ... (Test logic remains largely same, just load student) ...
        # For brevity, implementing a simple test loop if needed or user runs choice=1 for now.
        # But let's check original logic. It had choice 2.
        
        test_img_paths = np.array(data_paths['test']['img_paths'])
        test_cls_ids = np.array(data_paths['test']['cls_ids'])
        test_loader = get_dataloader(cfg, split_type='test', img_paths=test_img_paths, cls_ids=test_cls_ids)
        
        results = []
        use_timm_flag = getattr(cfg, 'use_timm', True)

        for fold_id in range(cfg.num_folds):
             ckpt_path = os.path.join(cfg.output_dir, f"fold_{fold_id}", "model_best.pth")
             if not os.path.exists(ckpt_path): continue
             
             logger.info(f"Testing Fold {fold_id} Student...")
             model = create_model_safe(cfg.student_model, use_timm_flag, pretrained=False, num_classes=cfg.num_classes, logger=logger)
             model.load_state_dict(torch.load(ckpt_path, map_location=device))
             model.to(device)
             
             loss, acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
             logger.info(f"Fold {fold_id} Test Acc: {acc:.2f}% | Loss: {loss:.4f}")
             results.append({"fold": fold_id, "acc": acc, "loss": loss})
             
        logger.info(f"Final Test Results: {results}")

if __name__ == "__main__":
    main()
