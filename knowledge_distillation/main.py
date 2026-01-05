import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_distillation.kd_config import KDConfig
from knowledge_distillation.dataset import get_dataloader
from knowledge_distillation.utils import get_logger, set_seed, save_checkpoint, load_checkpoint
from knowledge_distillation.train import train_kd_one_epoch, validate, test

def main():
    # Only parsing 'config' updates if we want to support CLI overrides in the future without defining them all here.
    # Currently user requested to remove duplicate definitions.
    # We will instantiate Config directly.
    
    cfg = KDConfig()
    
    # Setup Logger
    log_file = os.path.join(cfg.output_dir, f"{cfg.experiment_name}.log")
    logger = get_logger(cfg.experiment_name, log_file)
    logger.info(f"Configuration: {cfg}")
    
    # Set Seed
    set_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Data Loaders
    train_loader = get_dataloader(cfg.data_dir, 'train', cfg.batch_size, cfg.num_workers)
    val_loader = get_dataloader(cfg.data_dir, 'validation', cfg.batch_size, cfg.num_workers)
    
    # Create Models
    logger.info(f"Creating Teacher Model: {cfg.teacher_model}")
    try:
        teacher_model = timm.create_model(cfg.teacher_model, pretrained=True, num_classes=cfg.num_classes)
        teacher_model.to(device)
        teacher_model.eval() # Teacher is always frozen/eval
        
        # Load Teacher Checkpoint if provided
        # For KD, we usually need a pretrained teacher. 
        # We can look for one in the output dir if standard naming is used, or rely on config.
        # But for now, we'll assume it's passed or defaults.
        # Since we removed CLI args, we rely on Config defaults or manual changes.
        # If user wants to pass teacher_checkpoint, they should add it to KDConfig.
        
        # Checking if 'teacher_checkpoint' exists in config (I need to add it to KDConfig if it's not there, 
        # or just handle it here if it was passed via CLI before but now lost).
        # The original code had `args.teacher_checkpoint`. Config doesn't have it by default in `__init__` in previous steps.
        # I should probably add `teacher_checkpoint` to KDConfig to be safe.
        
        if hasattr(cfg, 'teacher_checkpoint') and cfg.teacher_checkpoint and os.path.exists(cfg.teacher_checkpoint):
            logger.info(f"Loading Teacher weights from {cfg.teacher_checkpoint}")
            state_dict = load_checkpoint(cfg.teacher_checkpoint, device=cfg.device)
            teacher_model.load_state_dict(state_dict)
        else:
            logger.warning("No teacher checkpoint provided in config. Using pretrained/random weights (NOT RECOMMENDED for KD).")
            
    except Exception as e:
        logger.error(f"Failed to create teacher model: {e}")
        return

    logger.info(f"Creating Student Model: {cfg.student_model}")
    try:
        student_model = timm.create_model(cfg.student_model, pretrained=True, num_classes=cfg.num_classes)
        student_model.to(device)
    except Exception as e:
        logger.error(f"Failed to create student model: {e}")
        return

    if cfg.choice == 1:
        # TRAINING Loop
        logger.info("Starting KD Training...")
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.AdamW(student_model.parameters(), lr=cfg.learning_rate)
        
        best_acc = 0.0
        
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
            
            # Save Best Student Model
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(cfg.output_dir, "best_student_model.pth")
                save_checkpoint(student_model.state_dict(), save_path)
                logger.info(f"Saved new best student model with Acc: {best_acc:.4f}")
            
            # Save Latest Student Model
            save_path_last = os.path.join(cfg.output_dir, "last_student_model.pth")
            save_checkpoint(student_model.state_dict(), save_path_last)
            
        logger.info(f"KD Training Completed. Best Student Accuracy: {best_acc:.4f}")

    elif cfg.choice == 2:
        # TESTING Loop
        logger.info("Starting Student Testing...")
        
        # Determine Checkpoint Path from Config
        ckpt_name = f"{cfg.test_ckpt_type}_student_model.pth"
        ckpt_path = os.path.join(cfg.output_dir, ckpt_name)
            
        if os.path.exists(ckpt_path):
            logger.info(f"Loading student checkpoint: {ckpt_path}")
            state_dict = load_checkpoint(ckpt_path, device=cfg.device)
            student_model.load_state_dict(state_dict)
        else:
            logger.warning(f"Student checkpoint not found at {ckpt_path}. Testing with random/pretrained weights.")
            
        test_acc = test(student_model, val_loader, device, logger)
        logger.info(f"Final Student Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
