import os
import torch
import random
import numpy as np
import logging
import sys
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from pruning.dataset import NeuDetDataset

def create_fold_split_idx(cfg, img_paths, cls_ids):
    skf = StratifiedKFold(n_splits=cfg.num_folds)

    fold_idx_dict = {}
    for i, (train_idx, test_idx) in enumerate(skf.split(img_paths, cls_ids)):
        fold_idx_dict[i] = {'train': train_idx.tolist(), 'validation': test_idx.tolist()}
        
    return fold_idx_dict


def build_img_paths(cfg):
    data_paths = {
        'train': {'img_paths': [], 'cls_ids': []},
        'test': {'img_paths': [], 'cls_ids': []}, 
    }
    
    train_imgs_dir_path = os.path.join(cfg.data_dir, 'train', 'images')
    for dir_name in os.listdir(train_imgs_dir_path):
        cls_id = cfg.cls_name_id_map[dir_name]
        
        all_img_fnames = os.listdir(os.path.join(train_imgs_dir_path, dir_name))
        data_paths['train']['img_paths'] += [os.path.join(train_imgs_dir_path, dir_name, img_fname) for img_fname in all_img_fnames]
        data_paths['train']['cls_ids'] += [cls_id] * len(all_img_fnames)

    # -----------------------------------------------
    
    test_imgs_dir_path = os.path.join(cfg.data_dir, 'validation', 'images')
    for dir_name in os.listdir(test_imgs_dir_path):
        cls_id = cfg.cls_name_id_map[dir_name]
        
        all_img_fnames = os.listdir(os.path.join(test_imgs_dir_path, dir_name))
        data_paths['test']['img_paths'] += [os.path.join(test_imgs_dir_path, dir_name, img_fname) for img_fname in all_img_fnames]
        data_paths['test']['cls_ids'] += [cls_id] * len(all_img_fnames)
    
    return data_paths


def get_dataloader(cfg, split_type, img_paths, cls_ids):
    # For pruning/finetuning, we generally want shuffle for train
    shuffle = True if split_type == 'train' else False
    
    dataset = NeuDetDataset(img_paths, cls_ids)
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, 
                        num_workers=cfg.num_workers, pin_memory=True)

    return loader


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(cfg):
    """Creates a logger that writes to console and optionally to a file."""
    # Ensure log directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    log_file_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}.log")

    logger = logging.getLogger(cfg.experiment_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file_path:
        fh = logging.FileHandler(log_file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger


def calculate_accuracy(output, target):
    """Calculates the accuracy."""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def save_checkpoint(cfg, model=None, optimizer=None, scheduler=None, training_log=None, 
                fold_idx_dict=None, fold_id=0, suffix="", is_teacher=False):
    
    ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_id}')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    prefix = "teacher_" if is_teacher else ""

    if fold_idx_dict:
        # Fold dict is global, save in output_dir root
        save_path = os.path.join(cfg.output_dir, "fold_idx_dict.json")
        with open(save_path, 'w') as f:
            json.dump(fold_idx_dict, f, indent=4)

    if model:
        # For pruning, we sometimes save the full model object if needed, 
        # but standard checkpointing saves state_dict.
        # IF the model is a Pruned model structure that can't be easily rebuilt from config,
        # we might need torch.save(model).
        # HOWEVER, the User instruction says "Make the repo self sufficient... Ensure no messy code."
        # Standard PyTorch practice is state_dict. 
        # But torch-pruning often implies saving model structure.
        # Let's save state_dict for consistency with other modules.
        # If full model save is needed, caller should do it manually or we add a flag.
        # I'll stick to state_dict for "checkpointing" logic.
        save_path = os.path.join(ckpt_dir, f"{prefix}model_{suffix}.pth")
        torch.save(model.state_dict(), save_path)

    if optimizer:
        save_path = os.path.join(ckpt_dir, f"{prefix}optimizer_{suffix}.pth")
        torch.save(optimizer.state_dict(), save_path)

    if scheduler:
        save_path = os.path.join(ckpt_dir, f"{prefix}scheduler_{suffix}.pth")
        torch.save(scheduler.state_dict(), save_path)

    if training_log:
        save_path = os.path.join(ckpt_dir, f"{prefix}training_log.json")
        with open(save_path, 'w') as f:
            json.dump(training_log, f, indent=4)


def load_checkpoint(cfg, load_type: str, model=None, fold_id=0, suffix: str = "best"):
    # Simplified loader for pruning needs
    ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_id}')
    
    if load_type == "model":
        assert model is not None
        load_path = os.path.join(ckpt_dir, f"model_{suffix}.pth")
        model.load_state_dict(torch.load(load_path, map_location=cfg.device))
        return model
    
    # Add other types if needed
    return None

def visualize_training_log(cfg, training_log, fold_id=0):
    train_loss = training_log['loss']
    # If val loss exists? The engine tracks 'accuracy' and 'loss'.
    # Current engine structure: history = {'loss': [], 'accuracy': []} (Train stats)
    # Validation stats were just printed?
    # Wait, previous step I added validation logic to finetune.
    # Did I add it to history? NO. I just used it for Best Model.
    # So history only contains Train Loss/Acc.
    
    ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_id}')
    viz_dir = os.path.join(ckpt_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Plot train loss
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(viz_dir, 'train_loss.png'))
    plt.close()

    # Accuracy Plots
    train_acc = training_log['accuracy']
    
    plt.plot(train_acc)
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(viz_dir, 'train_acc.png'))
    plt.close()
