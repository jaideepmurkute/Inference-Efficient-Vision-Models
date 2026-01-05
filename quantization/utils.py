import os
import torch
import random
import numpy as np
import logging
import sys
import json
import copy
from torch.utils.data import DataLoader
from knowledge_distillation.dataset import NeuDetDataset
from teacher_training.utils import create_fold_split_idx, build_img_paths


def get_dataloader(cfg, split_type, img_paths, cls_ids):
    # Validation/Test loaders should not be shuffled for consistency in eval
    # For calibration (subset of train), shuffling is fine/good, but we control that when selecting indices
    shuffle = False
    
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
    
    # Clear existing handlers
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


def print_size_of_model(model):
    """Prints the size of the model in MB."""
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size_mb

def load_model(model_path, model, device='cpu'):
    """Robustly loads a model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise e
