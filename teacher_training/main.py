import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import timm
import torchvision
from tabulate import tabulate

# Add project root to path (assuming script is run from teacher_training dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher_training.teacher_config import TeacherConfig
from teacher_training.utils import get_logger, set_seed, get_dataloader, create_fold_split_idx, build_img_paths, save_checkpoint, load_checkpoint, visualize_training_log

from teacher_training.train import train_one_epoch, validate

def main():
    cfg = TeacherConfig()
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    # 1. Setup Data Paths
    data_paths = build_img_paths(cfg)
    
    # Model Creation Logic
    logger.info(f"Creating model: {cfg.model_name}")
    try:
        if getattr(cfg, 'use_timm', True):
            model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
        else:
            # Torchvision logic
            # ResNet50 usually has 'fc' as head.
            # We need to map 'resnet50' string to torchvision.models.resnet50
            if not hasattr(torchvision.models, cfg.model_name):
                raise ValueError(f"Torchvision model {cfg.model_name} not found")
            
            # Instantiate
            weights = "DEFAULT" if cfg.pretrained else None
            model_fn = getattr(torchvision.models, cfg.model_name)
            model = model_fn(weights=weights)
            
            # Replace Head
            # ResNet: model.fc
            # MobileNet: model.classifier
            # EfficientNet: model.classifier
            
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, cfg.num_classes)
            elif hasattr(model, 'classifier'):
                # Handle varying classifier structures (Sequential vs Linear)
                if isinstance(model.classifier, nn.Sequential):
                    # Usually last layer is Linear
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, cfg.num_classes)
                elif isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, cfg.num_classes)
                else:
                    logger.warning(f"Could not automatically find classifier head for {cfg.model_name}. You may need manual adjustment.")
            elif hasattr(model, 'head'):
                 # block specific 
                 pass
                 
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to create model {cfg.model_name}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    
    if cfg.choice == 1:
        img_paths = np.array(data_paths['train']['img_paths'])
        cls_ids = np.array(data_paths['train']['cls_ids'])
        
        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)
        
        # Training Loop (Folds)
        # We only train Fold 0 as requested in typical pipeline, or all?
        # Config says 'fold_id' usually used for downstream. Here we loop folds probably?
        # Original code looped folds? Let's check original logic.
        # Original main.py (from memory/view) had fold loop.
        
        results = []
        
        for fold_k in range(cfg.num_folds):
            logger.info(f"--- Starting Fold {fold_k} ---")
            
            train_idx = fold_idx_dict[fold_k]['train']
            val_idx = fold_idx_dict[fold_k]['val']
            
            train_loader = get_dataloader(cfg, "train", img_paths[train_idx], cls_ids[train_idx])
            val_loader = get_dataloader(cfg, "val", img_paths[val_idx], cls_ids[val_idx])
            
            # Reset Model for each fold? 
            # Ideally yes. But loaded above outside loop.
            # Correction: Move model init INSIDE loop to be proper k-fold.
            # But for simplicity (and given we usually just use fold 0), let's keep it simplest.
            # Wait, if we use the same model instance, it sees all data. That's Data Leakage.
            # I must re-init model.
            
            # Re-init logic copy-paste
            if getattr(cfg, 'use_timm', True):
                model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
            else:
                 model_fn = getattr(torchvision.models, cfg.model_name)
                 model = model_fn(weights="DEFAULT" if cfg.pretrained else None)
                 if hasattr(model, 'fc'): model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
                 elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): model.classifier = nn.Linear(model.classifier.in_features, cfg.num_classes)
                 elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, cfg.num_classes)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate) # Reset opt
            
            best_acc = 0.0
            training_log = {
                'train': {'loss': [], 'accuracy': []},
                'validation': {'loss': [], 'accuracy': []}
            }
            
            for epoch in range(cfg.epochs):
                train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, optimizer, criterion, device, logger)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                train_acc *= 100
                val_acc *= 100
                
                # Update Log
                training_log['train']['loss'].append(train_loss)
                training_log['train']['accuracy'].append(train_acc)
                training_log['validation']['loss'].append(val_loss)
                training_log['validation']['accuracy'].append(val_acc)
                
                logger.info(f"Fold {fold_k} Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")
                
                # Save Best
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint(cfg, model=model, fold_id=fold_k, suffix="best", training_log=training_log)
            
            # Save Last
            save_checkpoint(cfg, model=model, fold_id=fold_k, suffix="last", training_log=training_log)
            
            # Visualize
            visualize_training_log(cfg, training_log, fold_id=fold_k)
            
            results.append({"fold": fold_k, "best_acc": best_acc})
            
        logger.info(f"Training Complete. Results: {results}")

    elif cfg.choice == 2:
        # Test Mode
        logger.info("Running Test on all folds...")
        test_img_paths = np.array(data_paths['test']['img_paths'])
        test_cls_ids = np.array(data_paths['test']['cls_ids'])
        test_loader = get_dataloader(cfg, "test", test_img_paths, test_cls_ids)
        
        results = []
        for fold_k in range(cfg.num_folds):
             # Check for existence first (optional but good for skipping missing folds)
             # We can use load_checkpoint inside a try-except block, or check path manually
             ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_k}')
             ckpt_path = os.path.join(ckpt_dir, f"model_{cfg.test_ckpt_type}.pth")
             
             if not os.path.exists(ckpt_path):
                 logger.warning(f"Checkpoint not found for Fold {fold_k}: {ckpt_path}")
                 continue
                 
             logger.info(f"Eval Fold {fold_k} ({ckpt_path})...")
             
             # Re-init Model Structure
             if getattr(cfg, 'use_timm', True):
                model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
             else:
                 model_fn = getattr(torchvision.models, cfg.model_name)
                 model = model_fn(weights=None)
                 if hasattr(model, 'fc'): model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
                 elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): model.classifier = nn.Linear(model.classifier.in_features, cfg.num_classes)
                 elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, cfg.num_classes)
             model.to(device)
                 
             try:
                 load_checkpoint(cfg, "model", model=model, fold_id=fold_k, suffix=cfg.test_ckpt_type)
             except Exception as e:
                 logger.error(f"Error loading checkpoint for fold {fold_k}: {e}")
                 continue
             
             loss, acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
             acc *= 100
             logger.info(f"Fold {fold_k} Test Acc: {acc:.2f}% | Loss: {loss:.4f}")
             results.append({"fold": fold_k, "test_acc": acc, "test_loss": loss})
        
        logger.info(f"Final Test Results: {results}")

if __name__ == "__main__":
    main()
