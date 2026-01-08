
import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.q_config import QuantConfig
from quantization.utils import get_logger, set_seed, load_model, print_size_of_model, get_dataloader, build_img_paths, create_fold_split_idx


def get_model(model_name, num_classes, pretrained=False):
    """
    Local helper to load torchvision models.
    """
    if not hasattr(torchvision.models, model_name):
         raise ValueError(f"Torchvision model {model_name} not found")
    
    weights = "DEFAULT" if pretrained else None
    model_fn = getattr(torchvision.models, model_name)
    model = model_fn(weights=weights)
    
    # Replace Head
    if hasattr(model, 'fc'): # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'): # MobileNet/DenseNet/VGG
         if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
         elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    
    return model


def main():
    cfg = QuantConfig()
    
    # Setup Logger
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # Multi-Fold Loop
    # -------------------------------------------------------------
    data_paths = build_img_paths(cfg)
    
    # Global Data Arrays (for fold splitting)
    all_train_img_paths = np.array(data_paths['train']['img_paths'])
    all_train_cls_ids = np.array(data_paths['train']['cls_ids'])
    
    # Test Data (Global Test Set - same for all folds usually, or validation split?)
    # In this project context:
    # 'train' dir is used for CV (split into train/val folds).
    # 'validation' dir is used as a held-out test set? 
    # Wait, teacher_utils `create_fold_split_idx` splits the 'train_img_paths'.
    # So `data_paths['train']` is the source for CV.
    
    fold_idx_dict = create_fold_split_idx(cfg, all_train_img_paths, all_train_cls_ids)
    
    results = []

    for fold_id in range(cfg.num_folds):
        logger.info(f"\n{'='*40}\nStarting Quantization Fold {fold_id}/{cfg.num_folds - 1}\n{'='*40}")
        cfg.fold_id = fold_id
        
        # Define Fold Output Dir
        fold_dir = os.path.join(cfg.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        # -------------------------------------------------------------
        # 1. Load Model
        # -------------------------------------------------------------
        model = None
        if cfg.model_type == 'teacher':
            ckpt_path = os.path.join(cfg.teacher_exp_path, f"fold_{cfg.fold_id}", "model_best.pth")
            model = get_model(cfg.teacher_model, cfg.num_classes, pretrained=False)
            model = load_model(ckpt_path, model, device=device)
            
        elif cfg.model_type == 'student':
            ckpt_path = os.path.join(cfg.student_exp_path, f"fold_{cfg.fold_id}", "model_best.pth")
            model = get_model(cfg.student_model, cfg.num_classes, pretrained=False)
            model = load_model(ckpt_path, model, device=device)
            
        elif cfg.model_type == 'pruned':
            # Pruned model has custom architecture (channels removed).
            # We must load the Full Model Object.
            ckpt_path = os.path.join(cfg.pruning_exp_path, f"fold_{cfg.fold_id}", "pruned_model.pth")
            logger.info(f"Loading FULL pruned model from: {ckpt_path}")
            
            if not os.path.exists(ckpt_path):
                 # Fallback check?
                 fallback = os.path.join(cfg.pruning_exp_path, f"fold_{cfg.fold_id}", "model_best.pth")
                 if os.path.exists(fallback):
                      logger.warning(f"Main pruned model not found. Trying fallback: {fallback}")
                      ckpt_path = fallback
                 else:
                      logger.warning(f"Pruned model NOT found for fold {fold_id}. Skipping.")
                      continue

            try:
                 # Load the full model object
                 # PyTorch 2.6+ defaults to weights_only=True, which breaks full model loading.
                 # We trust our own checkpoint, so we disable it.
                 model = torch.load(ckpt_path, map_location=device, weights_only=False)
                 
                 if isinstance(model, dict):
                      raise ValueError("Loaded object is a dictionary (state_dict), but 'pruned' model requires Full Model object.")
            except Exception as e:
                 logger.error(f"Failed to load pruned model: {e}")
                 # raise e
                 continue
                 
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

        model.to(device)
        model.eval()
        logger.info(f"Model Loaded Successfully (Fold {fold_id}).")

        # Measure Baseline Size
        fp32_size = print_size_of_model(model)
        logger.info(f"FP32 Model Size: {fp32_size:.2f} MB")
        
        
        # -------------------------------------------------------------
        # 2. Prepare Data (for calibration and evaluation)
        # -------------------------------------------------------------
        
        # Use Test Set for Evaluation (Global Test Set)
        val_img_paths = np.array(data_paths['test']['img_paths'])
        val_cls_ids = np.array(data_paths['test']['cls_ids'])
        test_loader = get_dataloader(cfg, 'test', val_img_paths, val_cls_ids)
        
        # Use Subset of Train Set (from this fold) for Calibration
        train_idx = fold_idx_dict[fold_id]['train']
        
        # Limit calibration size (e.g., 200 images or few batches)
        # Using a fixed seed subset of the fold's train data
        calib_size = min(len(train_idx), 256) # 256 images
        calib_train_idx = train_idx[:calib_size]
        
        calib_loader = get_dataloader(cfg, 'train', 
                                      all_train_img_paths[calib_train_idx], 
                                      all_train_cls_ids[calib_train_idx])


        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # 3. Quantization Experiments (Static, Dynamic, FP16)
        # -------------------------------------------------------------
        logger.info(f"Starting Quantization Experiments for Fold {fold_id}...")
        
        # Define methods to test
        methods = ['static_int8', 'dynamic_int8', 'fp16']
        
        for method in methods:
            logger.info(f"--- Running {method} ---")
            
            # Reload fresh model for each method to avoid graph conflicts
            # (In-memory copy is safer than re-loading from disk if possible, but load_model is cheap)
            work_model = deepcopy(model) # Deepcopy the float model
            work_model.eval()
            
            q_model = None
            
            try:
                if method == 'static_int8':
                    # FX Graph Mode Static Quantization
                    # 1. Backend
                    backend = "qnnpack" 
                    torch.backends.quantized.engine = backend
                    
                    # 2. Config - Robust configuration for Fold 3 stability
                    from torch.ao.quantization import QConfig, QConfigMapping
                    from torch.ao.quantization.observer import (
                        MovingAverageMinMaxObserver,
                        PerChannelMinMaxObserver
                    )
                    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
                    
                    # Use MinMax observers which are more robust to outliers than Histogram
                    weight_observer = PerChannelMinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        ch_axis=0
                    )

                    activation_observer = MovingAverageMinMaxObserver.with_args(
                        dtype=torch.quint8,
                        qscheme=torch.per_tensor_affine,
                        averaging_constant=0.01
                    )

                    robust_qconfig = QConfig(
                        activation=activation_observer,
                        weight=weight_observer
                    )
                    
                    # Explicit mapping to ensure fusion works correctly
                    qconfig_mapping = (
                        QConfigMapping()
                        .set_global(robust_qconfig)
                        .set_object_type(torch.nn.Conv2d, robust_qconfig)
                        .set_object_type(torch.nn.Linear, robust_qconfig)
                        .set_object_type(torch.nn.ReLU, robust_qconfig)
                        .set_object_type(torch.nn.BatchNorm2d, robust_qconfig)
                    )
                    
                    # 3. Prepare
                    example_inputs = (torch.randn(1, 3, cfg.image_size[0], cfg.image_size[1]),)
                    if device.type != 'cpu':
                         work_model.cpu() # FX requires CPU usually
                         
                    prepared_model = prepare_fx(work_model, qconfig_mapping, example_inputs)
                    
                    # 4. Calibrate
                    with torch.no_grad():
                        for images, labels in calib_loader:
                            images = images.to('cpu')
                            prepared_model(images)
                    
                    # 5. Convert
                    q_model = convert_fx(prepared_model)
                    
                elif method == 'dynamic_int8':
                    # Eager Mode Dynamic Quantization (Good for LSTM/Transformers, less for CNNs but worth testing)
                    # CNNs often don't benefit much from Dynamic Quant, but user requested it.
                    # FX also supports dynamic: get_default_qconfig_mapping("qnnpack") for dynamic?
                    # Let's use Eager mode for simplicity/diversity if compatible, OR FX Dynamic.
                    # FX Dynamic:
                    # qconfig_mapping = get_default_qconfig_mapping("qnnpack") -> usually static.
                    # For dynamic, we specify it manually or use Eager `quantize_dynamic`.
                    # Eager `quantize_dynamic` usually targets Linear/LSTM.
                    # Let's try Eager `torch.quantization.quantize_dynamic`.
                    # Note: This runs on CPU.
                    work_model.cpu()
                    q_model = torch.quantization.quantize_dynamic(
                        work_model, 
                        {nn.Linear}, # Only quantize Linear layers dynamically usually
                        dtype=torch.qint8
                    )
                    
                elif method == 'fp16':
                    # FP16 (Half Precision) - Requires GPU usually for speedup, or CPU on some hardware.
                    # Simple casting.
                    work_model.eval()
                    if device.type == 'cuda':
                        q_model = work_model.half()
                        q_model.to('cuda')
                    else:
                        # On CPU, FP16 inference might be slow or not supported natively for all ops.
                        # But valid for size reduction check.
                        q_model = work_model.half()
                
                # Measure Size
                # For FP16 on GPU, print_size needs to handle it.
                # print_size_of_model util saves to disk.
                q_size = print_size_of_model(q_model)
                logger.info(f"{method} Size: {q_size:.2f} MB | Reduction: {fp32_size/q_size:.2f}x")
                
                # Evaluate Accuracy
                # Note: FP16 needs CUDA inputs if model is CUDA.
                eval_device = 'cuda' if (method == 'fp16' and device.type == 'cuda') else 'cpu'
                if eval_device == 'cpu': q_model.cpu()
                
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(eval_device)
                        labels = labels.to(eval_device)
                        
                        if method == 'fp16':
                            images = images.half()
                            
                        outputs = q_model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                acc = 100 * correct / total
                logger.info(f"{method} Accuracy: {acc:.2f}%")
                
                results.append({
                    'Fold': fold_id,
                    'Method': method,
                    'FP32 Size (MB)': fp32_size,
                    'Quant Size (MB)': q_size,
                    'Reduction': fp32_size/q_size,
                    'Accuracy': acc
                })
                
                # Save Model (Only best or all? Saving all might take space)
                save_name = f"model_{method}.pth"
                save_path = os.path.join(fold_dir, save_name)
                torch.save(q_model.state_dict(), save_path)
            
            except Exception as e:
                logger.error(f"Quantization method {method} failed: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if results:
        import pandas as pd
        from tabulate import tabulate
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print(f"QUANTIZATION RESULTS ({cfg.model_type})")
        print("="*60)
        print(tabulate(df, headers='keys', tablefmt='grid'))
        
        avg_acc = df['Accuracy'].mean()
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")
        
        # Save Summary CSV
        df.to_csv(os.path.join(cfg.output_dir, "quantization_summary.csv"), index=False)


if __name__ == "__main__":
    main()
