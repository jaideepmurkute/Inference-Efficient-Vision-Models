import sys
import os
import torch
import numpy as np
import pandas as pd
import timm
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.q_config import QuantConfig
from quantization.utils import *
from quantization.engines import QuantizationEngine

def main():
    cfg = QuantConfig()
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    
    set_seed(cfg.seed)
    
    # 1. Setup Data
    # For quantization, we mostly care about CPU inference.
    device = torch.device('cpu') 
    
    # Build paths to find test data
    data_paths = build_img_paths(cfg)
    
    # We will use the 'test' (validation) split for evaluation
    img_paths = np.array(data_paths['test']['img_paths'])
    cls_ids = np.array(data_paths['test']['cls_ids'])
    
    # Use Fold 0 for consistency if we were doing CV, but for test set it logic is same
    # But for calibration (static quant), we need a subset of TRAIN data.
    train_img_paths = np.array(data_paths['train']['img_paths'])
    train_cls_ids = np.array(data_paths['train']['cls_ids'])
    
    # Stratified Split to get Fold 0 indices
    fold_idx_dict = create_fold_split_idx(cfg, train_img_paths, train_cls_ids)
    
    # Eval Loader (Full Validation Set)
    val_loader = get_dataloader(cfg, 'validation', img_paths, cls_ids)
    
    # Calibration Loader (Subset of Train Fold 0)
    # We take the first 'num_calibration_batches * batch_size' samples for calibration
    calib_indices = fold_idx_dict[cfg.fold_id]['train'][:cfg.num_calibration_batches * cfg.batch_size]
    calib_img_paths = train_img_paths[calib_indices]
    calib_cls_ids = train_cls_ids[calib_indices]
    
    calib_loader = get_dataloader(cfg, 'train', calib_img_paths, calib_cls_ids)
    
    
    # 2. Results Container
    results = []

    # 3. Load Baseline Model (Float32)
    logger.info(f"Loading Base {cfg.model_type.upper()} Model from Fold {cfg.fold_id}...")
    
    if cfg.model_type == 'teacher':
        model_name = cfg.teacher_model
        ckpt_path = os.path.join(cfg.teacher_exp_path, f"fold_{cfg.fold_id}", "model_best.pth")
        # teacher outputs are usually prefix free in filename if coming from main.py of teacher, 
        # but check utils.py in teacher_training. it saves as f"{cfg.experiment_name}_model_{suffix}.pth" 
        # actually wait, teacher_training main.py usage of save_checkpoint creates:
        # os.path.join(ckpt_dir, f"model_{suffix}.pth") -> fold_0/model_best.pth. Correct.
    else:
        model_name = cfg.student_model
        ckpt_path = os.path.join(cfg.student_exp_path, f"fold_{cfg.fold_id}", "model_best.pth")
    
    
    # Create Model Skeleton
    base_model = timm.create_model(model_name, pretrained=False, num_classes=cfg.num_classes)
    base_model.to(device)
    
    # Load Weights
    base_model = load_model(ckpt_path, base_model, device)
    base_model.eval()
    
    # Engine
    engine = QuantizationEngine(logger)
    
    # --- BASELINE EVALUATION ---
    logger.info("Evaluating FP32 Baseline...")
    
    fp32_size = print_size_of_model(base_model)
    fp32_acc = engine.evaluate_accuracy(base_model, val_loader, device)
    
    # For latency, use a dummy input
    dummy_input = torch.randn(1, 3, *cfg.image_size).to(device)
    fp32_lat = engine.measure_latency(base_model, dummy_input)
    
    logger.info(f"FP32 :: Size: {fp32_size:.2f} MB | Acc: {fp32_acc:.2f}% | Latency: {fp32_lat:.4f} ms")
    
    results.append({
        "Model": "FP32 Baseline",
        "Size (MB)": fp32_size,
        "Accuracy (%)": fp32_acc,
        "Latency (ms)": fp32_lat
    })
    
    # --- EXPERIMENT 1: DYNAMIC QUANTIZATION ---
    logger.info("--- Starting Dynamic Quantization ---")
    '''
    FP32 input
       ↓ (dynamic quantization)
    INT8 activations
        ×
    INT8 weights
        ↓
    INT32 accumulate
        ↓ (rescale; with scaling parameters in float32)
    FP32 output

    
    Accumulation happens in int32 because of:
        Hardware needs
        To avoid frequent overflow causing quantization noise
    Final output is in float32  because of:
        Scaling parameters are in float32 to retain precision
        To avoid loss of precision in final output
    '''
    dyn_model = engine.dynamic_quantize(base_model)
    
    dyn_size = print_size_of_model(dyn_model)
    dyn_acc = engine.evaluate_accuracy(dyn_model, val_loader, device)
    dyn_lat = engine.measure_latency(dyn_model, dummy_input)
    
    logger.info(f"Dynamic INT8 :: Size: {dyn_size:.2f} MB | Acc: {dyn_acc:.2f}% | Latency: {dyn_lat:.4f} ms")
    
    results.append({
        "Model": "Dynamic INT8",
        "Size (MB)": dyn_size,
        "Accuracy (%)": dyn_acc,
        "Latency (ms)": dyn_lat
    })
    
    # Save Dynamic Model
    save_path = os.path.join(cfg.output_dir, f"dynamic_quant_{cfg.model_type}_fold{cfg.fold_id}.pth")
    torch.save(dyn_model.state_dict(), save_path)
    
    
    # --- EXPERIMENT 2: STATIC QUANTIZATION ---
    logger.info("--- Starting Static Quantization ---")
    
    # For ViT, Static Quantization support in PyTorch is tricky via standard Eager Mode 
    # because it requires Quant/DeQuant stubs and fusing. 
    # However, we will try the standard flow. If it fails due to unsupported operations 
    # (common with LayerNorm/GELU in older PyTorch), we validly catch it.
    # Note: ViT Static Quant often requires 'FX Graph Mode' quantization, which is more advanced.
    # We will attempt Eager Mode here as a baseline "Skill Showcase".
    
    try:
        stat_model = engine.static_quantize(base_model, calib_loader)
        
        stat_size = print_size_of_model(stat_model)
        stat_acc = engine.evaluate_accuracy(stat_model, val_loader, device)
        stat_lat = engine.measure_latency(stat_model, dummy_input)
        
        logger.info(f"Static INT8 :: Size: {stat_size:.2f} MB | Acc: {stat_acc:.2f}% | Latency: {stat_lat:.4f} ms")
        
        results.append({
            "Model": "Static INT8",
            "Size (MB)": stat_size,
            "Accuracy (%)": stat_acc,
            "Latency (ms)": stat_lat
        })
        
        # Save Static Model
        save_path = os.path.join(cfg.output_dir, f"static_quant_{cfg.model_type}_fold{cfg.fold_id}.pth")
        torch.save(stat_model.state_dict(), save_path)
        
    except Exception as e:
        logger.error(f"Static Quantization Failed (Common with ViT in Eager Mode): {e}")
        logger.warning("To properly Static Quantize ViT, one often needs FX Graph Mode or ONNX Runtime.")
        results.append({
            "Model": "Static INT8",
            "Size (MB)": "N/A",
            "Accuracy (%)": "Failed",
            "Latency (ms)": "Failed (See Log)"
        })

    # --- REPORTING ---
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"QUANTIZATION RESULTS ({cfg.model_type.upper()} - Fold {cfg.fold_id})")
    print("="*50)
    print(tabulate(df, headers='keys', tablefmt='grid'))
    # Save CSV
    csv_path = os.path.join(cfg.output_dir, "quantization_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
