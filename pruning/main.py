import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.p_config import PruningConfig
from pruning.pruning_engine_structured import StructuredPruningEngine
from pruning.utils import get_logger, set_seed, get_dataloader, create_fold_split_idx, build_img_paths, save_checkpoint


def get_model(model_name, num_classes, pretrained=False):
    """
    Simpler model loader for torchvision only.
    """
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
             print(f"Warning: classifier head structure unknown for {model_name}")
    
    return model


def main():
    cfg = PruningConfig()
    
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    logger.info(f"Pruning Type: {getattr(cfg, 'pruning_type', 'structured')}")
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    # 1. Setup Data using local utils and dataset
    data_paths = build_img_paths(cfg)
    
    train_img_paths = np.array(data_paths['train']['img_paths'])
    train_cls_ids = np.array(data_paths['train']['cls_ids'])
    
    # Validation data (Global Test Set)
    val_img_paths = np.array(data_paths['test']['img_paths'])
    val_cls_ids = np.array(data_paths['test']['cls_ids'])
    
    # Get Fold Indices for all folds
    fold_idx_dict = create_fold_split_idx(cfg, train_img_paths, train_cls_ids)
    
    # ---------------------------------------------------------
    # Multi-Fold Loop
    # ---------------------------------------------------------
    test_loader = get_dataloader(cfg, 'test', val_img_paths, val_cls_ids)

    if cfg.choice == 1:
        # Full Pruning Pipeline
        for fold_id in range(cfg.num_folds):
            logger.info(f"\n{'='*40}\nStarting Fold {fold_id}/{cfg.num_folds - 1}\n{'='*40}")
            cfg.fold_id = fold_id
            
            # Define paths
            fold_dir = os.path.join(cfg.output_dir, f"fold_{fold_id}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # 1. Load Pre-trained Student (Source)
            # We look for "model_best.pth" in the student output directory
            ckpt_path = os.path.join(cfg.student_exp_path, f"fold_{fold_id}", "model_best.pth")
            
            if not os.path.exists(ckpt_path):
                 logger.warning(f"Checkpoint not found at {ckpt_path}. Skipping Fold {fold_id}.")
                 continue
            
            # Create Model Skeleton
            model = get_model(cfg.model_name, cfg.num_classes, pretrained=False)
            model.to(device)
            
            # Load Weights
            logger.info(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            
            # Handle state dict mismatch (e.g. if saved as nested dict) - Standard defensive check
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            # Clean prefix if needed
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): new_state[k[7:]] = v
                else: new_state[k] = v
                
            model.load_state_dict(new_state, strict=False)
            
            
            # 2. Initialize Engine
            engine = StructuredPruningEngine(cfg, logger)
            results = []
            
            # --- PHASE 1: BASELINE ---
            logger.info("Evaluating Baseline...")
            base_metrics = engine.evaluate_metrics(model, test_loader)
            base_metrics['Stage'] = 'Baseline'
            results.append(base_metrics)
            logger.info(f"Baseline: {base_metrics}")
            
            
            # --- PHASE 2: PRUNING ---
            # Note: torch-pruning modifies model in-place.
            # We save a copy of the structure? usually unnecessary if we save full model later.
            model = engine.prune_model(model)
            
            # Evaluate immediately (No FT)
            pruned_metrics = engine.evaluate_metrics(model, test_loader)
            pruned_metrics['Stage'] = 'Pruned (No FT)'
            results.append(pruned_metrics)
            logger.info(f"Pruned (No FT): {pruned_metrics}")
            
            
            # --- PHASE 3: FINE-TUNING ---
            logger.info("--- Phase 3: Fine-tuning ---")
            
            # Prepare Train Loader for this fold
            train_loader = get_dataloader(cfg, 'train', 
                                          train_img_paths[fold_idx_dict[fold_id]['train']], 
                                          train_cls_ids[fold_idx_dict[fold_id]['train']])
            
            # Note: We pass train_loader (fold specific) and test_loader (global validation)
            # Engine now returns (model, history)
            model, history = engine.finetune(model, train_loader, test_loader, cfg.finetune_epochs, cfg.learning_rate)
            
            # Save Training Log using local util
            save_checkpoint(cfg, model=None, training_log=history, fold_id=fold_id, suffix="pruning")
            
            # Final Evaluation
            final_metrics = engine.evaluate_metrics(model, test_loader)
            final_metrics['Stage'] = 'Pruned + FT'
            results.append(final_metrics)
            logger.info(f"Final: {final_metrics}")
            
            
            # --- REPORTING ---
            df = pd.DataFrame(results)
            cols = ['Stage', 'Accuracy', 'Latency (ms)', 'MACs (G)', 'Params (M)', 'Size (MB)']
            df = df[cols]
            
            print("\n" + "="*80)
            print(f"PRUNING RESULTS (Fold {fold_id} | {cfg.model_name} -> Ratio: {cfg.pruning_ratio})")
            print("="*80)
            print(tabulate(df, headers='keys', tablefmt='grid'))
            
            # Save Model
            # For pruned models, saving the full model object is safer to preserve the structure (channels removed)
            save_path = os.path.join(fold_dir, "pruned_model.pth") 
            torch.save(model, save_path) 
            logger.info(f"Pruned model (full structure) saved to {save_path}")
            
            # Save CSV
            csv_path = os.path.join(fold_dir, "results.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")

    elif cfg.choice == 2:
        # Pure Test Mode
        logger.info(f"Running Pure Test Mode (choice=2) on {cfg.num_folds} folds...")
        
        # Initialize Engine (for evaluation methods)
        engine = StructuredPruningEngine(cfg, logger)
        test_results = []

        for fold_id in range(cfg.num_folds):
            logger.info(f"Evaluating Fold {fold_id}...")
            
            fold_dir = os.path.join(cfg.output_dir, f"fold_{fold_id}")
            model_path = os.path.join(fold_dir, "pruned_model.pth")
            
            if not os.path.exists(model_path):
                 logger.warning(f"Pruned model not found: {model_path}. Skipping.")
                 continue
                 
            try:
                # Load Full Model Object
                model = torch.load(model_path, map_location=device, weights_only=False)
                model.to(device)
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                continue
                
            metrics = engine.evaluate_metrics(model, test_loader)
            metrics['Fold'] = fold_id
            test_results.append(metrics)
            logger.info(f"Fold {fold_id} Results: {metrics}")
            
        # Summary
        if test_results:
            df = pd.DataFrame(test_results)
            cols = ['Fold', 'Accuracy', 'Latency (ms)', 'MACs (G)', 'Params (M)']
            # Filter cols that exist
            cols = [c for c in cols if c in df.columns] 
            df = df[cols]
            
            print("\n" + "="*80)
            print(f"TEST MODE RESULTS ({cfg.model_name})")
            print("="*80)
            print(tabulate(df, headers='keys', tablefmt='grid'))
            
            save_csv = os.path.join(cfg.output_dir, "test_results.csv")
            df.to_csv(save_csv, index=False)
            logger.info(f"Test results saved to {save_csv}")
        else:
            logger.warning("No results collected in Test Mode.")

if __name__ == "__main__":
    main()
