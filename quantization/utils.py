import os
import random
import logging
import sys

import torch
from torch.utils.data import DataLoader

import numpy as np

from quantization.dataset import NeuDetDataset

# ------------------------------------------------------------------------


def create_fold_split_idx(cfg, img_paths, cls_ids):
    """Create fold split indices for stratified CV."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    fold_idx_dict = {}
    for i, (train_idx, test_idx) in enumerate(skf.split(img_paths, cls_ids)):
        fold_idx_dict[i] = {
            "train": train_idx.tolist(),
            "validation": test_idx.tolist(),
        }

    return fold_idx_dict


def build_img_paths(cfg):
    data_paths = {
        "train": {"img_paths": [], "cls_ids": []},
        "test": {"img_paths": [], "cls_ids": []},
    }

    train_imgs_dir_path = os.path.join(cfg.data_dir, "train", "images")
    for dir_name in os.listdir(train_imgs_dir_path):
        cls_id = cfg.cls_name_id_map[dir_name]

        all_img_fnames = os.listdir(os.path.join(train_imgs_dir_path, dir_name))
        data_paths["train"]["img_paths"] += [
            os.path.join(train_imgs_dir_path, dir_name, img_fname)
            for img_fname in all_img_fnames
        ]
        data_paths["train"]["cls_ids"] += [cls_id] * len(all_img_fnames)

    # -----------------------------------------------

    test_imgs_dir_path = os.path.join(cfg.data_dir, "validation", "images")
    for dir_name in os.listdir(test_imgs_dir_path):
        cls_id = cfg.cls_name_id_map[dir_name]

        all_img_fnames = os.listdir(os.path.join(test_imgs_dir_path, dir_name))
        data_paths["test"]["img_paths"] += [
            os.path.join(test_imgs_dir_path, dir_name, img_fname)
            for img_fname in all_img_fnames
        ]
        data_paths["test"]["cls_ids"] += [cls_id] * len(all_img_fnames)

    return data_paths


def get_dataloader(cfg, split_type, img_paths, cls_ids):
    shuffle = False

    dataset = NeuDetDataset(img_paths, cls_ids)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return loader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    log_file_path = os.path.join(cfg.output_dir, f"{cfg.experiment_name}.log")

    logger = logging.getLogger(cfg.experiment_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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
    os.remove("temp.p")
    return size_mb


def load_model(model_path, model, device="cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        model.load_state_dict(new_state, strict=False)
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise e
