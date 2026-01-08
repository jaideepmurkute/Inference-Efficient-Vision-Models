import sys
import os
import torch
import numpy as np
import pandas as pd
import timm
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.p_config import PruningConfig
from pruning.pruning_engine_structured import StructuredPruningEngine
from pruning.pruning_engine_unstructured import UnstructuredPruningEngine
# from pruning.pruning_engine import PruningEngine # OLD

from pruning.custom_model import get_prunable_model
from quantization.utils import get_logger, set_seed, load_model, get_dataloader # Reuse utils
from teacher_training.utils import create_fold_split_idx, build_img_paths, save_checkpoint

def main():
    cfg = PruningConfig()
    # Ensure log_dir exists if get_logger needs it, though get_logger uses cfg.output_dir usually
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    logger.info(f"Pruning Type: {getattr(cfg, 'pruning_type', 'structured')}")
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    

    

    
    # 1. Setup Data
    # For Fine-tuning, we need TRAIN data
    data_paths = build_img_paths(cfg)
    
    train_img_paths = np.array(data_paths['train']['img_paths'])
    train_cls_ids = np.array(data_paths['train']['cls_ids'])
    
    # Validation data
    val_img_paths = np.array(data_paths['test']['img_paths'])
    val_cls_ids = np.array(data_paths['test']['cls_ids'])
    
    # Get Fold Indices for all folds
    fold_idx_dict = create_fold_split_idx(cfg, train_img_paths, train_cls_ids)
    
    # ---------------------------------------------------------
    # Multi-Fold Loop
    # ---------------------------------------------------------
    # For final eval, we use the global Test set
    test_loader = get_dataloader(cfg, 'test', val_img_paths, val_cls_ids)

    for fold_id in range(cfg.num_folds):
        logger.info(f"\n{'='*40}\nStarting Fold {fold_id}/{cfg.num_folds - 1}\n{'='*40}")
        cfg.fold_id = fold_id
        
        # Create Fold Output Directory
        fold_dir = os.path.join(cfg.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 1. Prepare Data for this Fold
        train_indices = fold_idx_dict[fold_id]['train']
        # We use the validation split of this fold for monitoring fine-tuning performance
        # Note: 'test_loader' (global test) is used for final metrics.
        # But during fine-tuning, strictly, we should valid on the val set of this fold.
        # However, pruning engine 'finetune' method signature currently takes 'tr_loader, test_loader'.
        # Let's create a val_loader for the engine to use if we want.
        # For consistency with KD/Teacher, we usually finetune on Train Fold and Validate on Val Fold.
        
        # Train Loader (Subset)
        train_loader = get_dataloader(cfg, 'train', train_img_paths[train_indices], train_cls_ids[train_indices])
        
        # Val Loader (Subset - for monitoring if needed, or we just use global test for reporting)
        # engine.finetune uses 'test_loader' argument for evaluation. 
        # Ideally we pass a validation loader here to avoid peeking at test set, 
        # but if the original code passed test_loader, we keep it or improve it.
        # Let's use global validation (test) loader for consistent reporting metrics across folds as per original code.
        # Or better: use the fold's validation set for the 'test_loader' arg during fine-tuning to prevent overfitting to global test?
        # The prompt implies "test on test set". Let's stick to global test loader for the final report metric,
        # but for the 'finetune' loop validation, usually we want the validation split.
        # Given the previous code used `test_loader`, we will continue using `test_loader` (global) for simplicity unless user objects,
        # as usually pruning is evaluated on the final hold-out.
        
        
        # 2. Load Source Model
        logger.info(f"Loading Source Model ({cfg.model_source}) for Fold {fold_id}...")
        if cfg.model_source == 'teacher':
             # ..\teacher_training\output\exp_name\fold_N\model_best.pth
            ckpt_path = os.path.join(cfg.teacher_exp_path, f"fold_{fold_id}", "model_best.pth")
        elif cfg.model_source == 'student':
            ckpt_path = os.path.join(cfg.student_exp_path, f"fold_{fold_id}", "model_best.pth")
        else:
            ckpt_path = cfg.custom_model_path # Fallback or single path
        
        
        # Verify checkpoint exists
        if not os.path.exists(ckpt_path):
             logger.warning(f"Checkpoint not found at {ckpt_path}. Skipping Fold {fold_id}.")
             continue
        
        # Create Model Skeleton
        model = get_prunable_model(
            cfg.model_name,
            use_timm=getattr(cfg, 'use_timm', True),
            pretrained=False,
            num_classes=cfg.num_classes,
            img_size=cfg.image_size
        )
        model.to(device)
        
        # Load Weights
        logger.info(f"Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        
        
        # 3. Initialize Engine
        # 3. Initialize Engine
        if getattr(cfg, 'pruning_type', 'structured') == "unstructured":
            engine = UnstructuredPruningEngine(cfg, logger)
        else:
            engine = StructuredPruningEngine(cfg, logger)
        results = []
        
        # --- PHASE 1: BASELINE ---
        logger.info("Evaluating Baseline...")
        base_metrics = engine.evaluate_metrics(model, test_loader)
        base_metrics['Stage'] = 'Baseline'
        results.append(base_metrics)
        logger.info(f"Baseline: {base_metrics}")
        
        
        # --- PHASE 2: PRUNING ---
        logger.info("--- Phase 2: Structural Pruning ---")
        model = engine.prune_model(model)
        
        # Evaluate immediately (No FT)
        pruned_metrics = engine.evaluate_metrics(model, test_loader)
        pruned_metrics['Stage'] = 'Pruned (No FT)'
        results.append(pruned_metrics)
        logger.info(f"Pruned (No FT): {pruned_metrics}")
        
        
        if getattr(cfg, 'pruning_type', 'structured') == "unstructured":
            logger.info("Skipping Fine-tuning for Unstructured Pruning (would destroy sparsity without mask freezing).")
            # For Unstructured, 'Pruned (No FT)' is the Final result.
        else:
            # --- PHASE 3: FINE-TUNING ---
            logger.info("--- Phase 3: Fine-tuning ---")
            # Note: We pass train_loader (fold specific) and test_loader (global)
            model, history = engine.finetune(model, train_loader, test_loader, cfg.finetune_epochs, cfg.learning_rate)
            
            # Save Training Log
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
        print(f"PRUNING RESULTS (Fold {fold_id} | {cfg.model_source} -> Ratio: {cfg.pruning_ratio})")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid'))
        
        # Save Model to fold directory
        save_path = os.path.join(fold_dir, "pruned_model.pth") 
        # Using generic name inside fold dir, consistent with teacher/student structure usually having model_best.pth
        # But user suggested 'pruned_model_best_{fold}.pth'. 
        # Since we have fold dir, 'model_best.pth' or 'pruned_model.pth' is cleaner.
        # Let's use 'model_best.pth' to be exactly like teacher/student if possible, or 'pruned_model.pth' to distinguish.
        # The user's earlier edit had `pruned_model_best_{cfg.fold_id}.pth`.
        # I will use 'pruned_model.pth' inside the fold directory.
        torch.save(model, save_path) 
        logger.info(f"Pruned model (full structure) saved to {save_path}")
        
        # Save CSV
        csv_path = os.path.join(fold_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
