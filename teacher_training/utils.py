import os
import logging
import sys

import torch
import random
import numpy as np
import json

from torch.utils.data import DataLoader
from teacher_training.dataset import NeuDetDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import timm


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
    shuffle = True if split_type == 'train' and cfg.choice == 1 else False
    
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
                fold_idx_dict=None, fold_id=0, suffix=""):
    
    if fold_idx_dict:
        save_path = os.path.join(cfg.output_dir, "fold_idx_dict.json")
        with open(save_path, 'w') as f:
            json.dump(fold_idx_dict, f, indent=4)
    
    ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_id}')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if model:
        save_path = os.path.join(ckpt_dir, f"model_{suffix}.pth")
        torch.save(model.state_dict(), save_path)

    if optimizer:
        save_path = os.path.join(ckpt_dir, f"optimizer_{suffix}.pth")
        torch.save(optimizer.state_dict(), save_path)

    if scheduler:
        save_path = os.path.join(ckpt_dir, f"scheduler_{suffix}.pth")
        torch.save(scheduler.state_dict(), save_path)

    if training_log:
        save_path = os.path.join(ckpt_dir, "training_log.json")
        with open(save_path, 'w') as f:
            json.dump(training_log, f, indent=4)

    

def load_checkpoint(cfg, load_type: str, model=None, optimizer=None, scheduler=None, fold_id=0, suffix: str = "best"):
    valid_load_types = ["model", "optimizer", "scheduler", "training_log"]
    assert load_type in valid_load_types, f"Invalid load_type. Must be one of {valid_load_types}"
    
    if load_type == "fold_idx_dict":
        load_path = os.path.join(cfg.output_dir, "fold_idx_dict.json")
        try:
            with open(load_path, 'r') as f:
                fold_idx_dict = json.load(f)
            return fold_idx_dict
        except Exception as e:
            print(f"Failed to load fold_idx_dict: {e}")
    
    ckpt_dir = os.path.join(cfg.output_dir, f'fold_{fold_id}')

    if load_type == "model":
        assert model is not None, "Model object must be provided to load model checkpoint into."
        load_path = os.path.join(ckpt_dir, f"model_{suffix}.pth")
        try:
            model.load_state_dict(torch.load(load_path))
            return model
        except Exception as e:
            print(f"Failed to load model checkpoint at: {load_path}")
            raise e
        
    if load_type == "optimizer":
        assert optimizer is not None, "Optimizer object must be provided to load optimizer checkpoint into."
        load_path = os.path.join(ckpt_dir, f"optimizer_{suffix}.pth")
        try:
            optimizer.load_state_dict(torch.load(load_path))
            return optimizer
        except Exception as e:
            print(f"Failed to load optimizer checkpoint: {e}")

    if load_type == "scheduler":
        assert scheduler is not None, "Scheduler object must be provided to load scheduler checkpoint into."
        load_path = os.path.join(ckpt_dir, f"scheduler_{suffix}.pth")
        try:
            scheduler.load_state_dict(torch.load(load_path))
            return scheduler
        except Exception as e:
            print(f"Failed to load scheduler checkpoint: {e}")

    if load_type == "training_log":
        load_path = os.path.join(ckpt_dir, "training_log.json")
        try:
            with open(load_path, 'r') as f:
                training_log = json.load(f)
            return training_log
        except Exception as e:
            print(f"Failed to load training log: {e}")

    

def visualize_training_log(cfg, training_log, fold_id=0):
    train_loss = training_log['train']['loss']
    val_loss = training_log['validation']['loss']
    
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

    # Plot val loss
    plt.plot(val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(viz_dir, 'val_loss.png'))
    plt.close()
    
    # Plot both loss in same plot
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(viz_dir, 'train_val_loss.png'))
    plt.close()

    #----------------------------------------------------------------------
    
    # Accuracy Plots
    val_acc = training_log['validation']['accuracy']
    train_acc = training_log['train']['accuracy']
    
    # Plot train accuracy
    plt.plot(train_acc)
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(viz_dir, 'train_acc.png'))
    plt.close()
    # Plot val accuracy
    plt.plot(val_acc)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(viz_dir, 'val_acc.png'))
    plt.close()
    # Plot both accuracy in same plot
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(viz_dir, 'train_val_acc.png'))
    plt.close()
    
    #----------------------------------------------------------------------

