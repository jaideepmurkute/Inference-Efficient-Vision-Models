import argparse
import sys
import os

import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import numpy as np


# Add project root to path to allow imports if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher_training.teacher_config import TeacherConfig
from teacher_training.utils import *
from teacher_training.train import train_one_epoch, validate, test


def main():
    # Only parsing 'config' updates if we want to support CLI overrides in the future without defining them all here.
    # Currently user requested to remove duplicate definitions.
    # We will instantiate Config directly.
    
    cfg = TeacherConfig()
    
    # Setup Logger
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
        
    # Set Seed
    set_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    
    data_paths = build_img_paths(cfg)
    
    # Model - using timm
    logger.info(f"Creating model: {cfg.model_name}")
    try:
        model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to create model {cfg.model_name}: {e}")
        return

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    if cfg.choice == 1:
        img_paths = np.array(data_paths['train']['img_paths'])
        cls_ids = np.array(data_paths['train']['cls_ids'])
        # print("img_paths", img_paths)
        # print("cls_ids", cls_ids)
        # raise

        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)

        for fold_id in range(cfg.num_folds):
            print("Training Fold", fold_id)
            train_loader = get_dataloader(cfg, split_type='train', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['train']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['train']])
            
            val_loader = get_dataloader(cfg, split_type='validation', 
                                img_paths=img_paths[fold_idx_dict[fold_id]['validation']], 
                                cls_ids=cls_ids[fold_idx_dict[fold_id]['validation']])

            # TRAINING Loop
            logger.info("Starting Training...")
            best_acc = float('-inf')
            
            # Initialize dictionary to store training history
            training_log = {
                'train': {'loss': [], 'accuracy': []},
                'validation': {'loss': [], 'accuracy': []},
                'epoch_times': []
            }
            
            for epoch in range(cfg.epochs):
                train_loss, train_acc, epoch_time = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, logger
                )
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                logger.info(
                    f"Epoch [{epoch+1}/{cfg.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Update training log
                training_log['train']['loss'].append(train_loss)
                training_log['train']['accuracy'].append(train_acc)
                training_log['validation']['loss'].append(val_loss)
                training_log['validation']['accuracy'].append(val_acc)
                training_log['epoch_times'].append(epoch_time)
                
                # Save Best Model
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(cfg, model=model, fold_id=fold_id, suffix="best")
                    logger.info(f"Saved new best model with Acc: {best_acc:.4f}")
                
                # Save Latest Model
                save_checkpoint(cfg, model=model, fold_id=fold_id, suffix="last")
                logger.info(f"Saved latest model")

            logger.info(f"Training Completed. Best Validation Accuracy: {best_acc:.4f}")

            # Save training log to JSON
            save_checkpoint(cfg, fold_id=fold_id, training_log=training_log)
            logger.info(f"Saved training log to JSON")
                
            visualize_training_log(cfg, training_log, fold_id=fold_id)
            logger.info("Training Completed. Visualized Training Log")

            print('-' * 50)

    elif cfg.choice == 2:
        # TESTING Loop
        logger.info("Starting Testing...")
        
        data_type = 'test' # train / validation / test
        
        if data_type in ['train', 'validation']:
            img_paths = np.array(data_paths[data_type]['img_paths'])
            cls_ids = np.array(data_paths[data_type]['cls_ids'])

            fold_idx_dict = load_checkpoint(cfg, load_type='fold_idx_dict')

            for fold_id in fold_idx_dict.keys():
                img_paths = img_paths[fold_idx_dict[fold_id][data_type]]
                cls_ids = cls_ids[fold_idx_dict[fold_id][data_type]]
                
                print("Evaluating Fold", fold_id)
                data_loader = get_dataloader(cfg, split_type=data_type, img_paths=img_paths, 
                                            cls_ids=cls_ids)

                model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
                model.to(device)
                model = load_checkpoint(cfg, load_type='model', model=model, suffix=cfg.test_ckpt_type)
                
                test_acc = test(model, data_loader, device, logger)
                
                logger.info(f"Fold {fold_id} {data_type} Accuracy: {test_acc:.4f}")
                
        elif data_type == 'test':
            img_paths = np.array(data_paths[data_type]['img_paths'])
            cls_ids = np.array(data_paths[data_type]['cls_ids'])
            data_loader = get_dataloader(cfg, split_type=data_type, img_paths=img_paths, 
                                        cls_ids=cls_ids)

            model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
            model.to(device)
            model = load_checkpoint(cfg, load_type='model', model=model, suffix=cfg.test_ckpt_type)
            
            test_acc = test(model, data_loader, device, logger)
            
            logger.info(f"{data_type} Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    
    main()
