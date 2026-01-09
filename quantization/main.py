import sys
import os
from copy import deepcopy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision

from quantization.q_config import QuantConfig
from quantization.utils import (
    get_logger,
    set_seed,
    load_model,
    print_size_of_model,
    get_dataloader,
    build_img_paths,
    create_fold_split_idx,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------


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
    if hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):  # MobileNet/DenseNet/VGG
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    return model


def main():
    cfg = QuantConfig()

    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    data_paths = build_img_paths(cfg)

    all_train_img_paths = np.array(data_paths["train"]["img_paths"])
    all_train_cls_ids = np.array(data_paths["train"]["cls_ids"])

    fold_idx_dict = create_fold_split_idx(cfg, all_train_img_paths, all_train_cls_ids)

    results = []

    for fold_id in range(cfg.num_folds):
        logger.info(
            f"\n{'=' * 40}\nStarting Quantization Fold {fold_id}/{cfg.num_folds - 1}\n{'=' * 40}"
        )
        cfg.fold_id = fold_id

        fold_dir = os.path.join(cfg.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        model = None
        if cfg.model_type == "teacher":
            ckpt_path = os.path.join(
                cfg.teacher_exp_path, f"fold_{cfg.fold_id}", "model_best.pth"
            )
            model = get_model(cfg.teacher_model, cfg.num_classes, pretrained=False)
            model = load_model(ckpt_path, model, device=device)

        elif cfg.model_type == "student":
            ckpt_path = os.path.join(
                cfg.student_exp_path, f"fold_{cfg.fold_id}", "model_best.pth"
            )
            model = get_model(cfg.student_model, cfg.num_classes, pretrained=False)
            model = load_model(ckpt_path, model, device=device)

        elif cfg.model_type == "pruned":
            # Pruned model may have custom architecture (channels removed) - \
            # based on pruning method. Load the Full Model Object.
            ckpt_path = os.path.join(
                cfg.pruning_exp_path, f"fold_{cfg.fold_id}", "pruned_model.pth"
            )
            logger.info(f"Loading FULL pruned model from: {ckpt_path}")

            if not os.path.exists(ckpt_path):
                fallback = os.path.join(
                    cfg.pruning_exp_path, f"fold_{cfg.fold_id}", "model_best.pth"
                )
                if os.path.exists(fallback):
                    logger.warning(
                        f"Main pruned model not found. Trying fallback: {fallback}"
                    )
                    ckpt_path = fallback
                else:
                    logger.warning(
                        f"Pruned model NOT found for fold {fold_id}. Skipping."
                    )
                    continue

            try:
                model = torch.load(ckpt_path, map_location=device, weights_only=False)

                if isinstance(model, dict):
                    raise ValueError(
                        "Loaded object is a dictionary (state_dict), but 'pruned' model requires Full Model object."
                    )
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

        # Prepare data for calibration and evaluation
        # Use test set for evaluation
        val_img_paths = np.array(data_paths["test"]["img_paths"])
        val_cls_ids = np.array(data_paths["test"]["cls_ids"])
        test_loader = get_dataloader(cfg, "test", val_img_paths, val_cls_ids)

        # Use subset of fold's train set for calibration
        train_idx = fold_idx_dict[fold_id]["train"]

        # Limit calibration size (e.g., 200 images or few batches)
        # Using a fixed seed subset of the fold's train data
        calib_size = min(len(train_idx), 256)  # 256 images
        calib_train_idx = train_idx[:calib_size]

        calib_loader = get_dataloader(
            cfg,
            "train",
            all_train_img_paths[calib_train_idx],
            all_train_cls_ids[calib_train_idx],
        )

        # -------------------------------------------------------------
        logger.info("Starting Quantization ...")
        logger.info(f"Fold: {fold_id}")

        # Define methods to test
        methods = ["static_int8", "dynamic_int8", "fp16"]

        for method in methods:
            logger.info(f"--- Running {method} ---")

            # Reload fresh model for each method to avoid graph conflicts
            # (In-memory copy is safer than re-loading from disk if possible, but load_model is cheap)
            work_model = deepcopy(model)  # Deepcopy the float model
            work_model.eval()

            q_model = None

            try:
                if method == "static_int8":
                    # FX Graph Mode Static Quantization
                    backend = "qnnpack"
                    torch.backends.quantized.engine = backend

                    from torch.ao.quantization import QConfig, QConfigMapping
                    from torch.ao.quantization.observer import (
                        MovingAverageMinMaxObserver,
                        PerChannelMinMaxObserver,
                    )
                    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

                    # MinMax observers - more robust to outliers than Histogram
                    weight_observer = PerChannelMinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_channel_symmetric,
                        ch_axis=0,
                    )

                    activation_observer = MovingAverageMinMaxObserver.with_args(
                        dtype=torch.quint8,
                        qscheme=torch.per_tensor_affine,
                        averaging_constant=0.01,
                    )

                    robust_qconfig = QConfig(
                        activation=activation_observer, weight=weight_observer
                    )

                    # Explicit mapping to ensure fusion working correctly
                    qconfig_mapping = (
                        QConfigMapping()
                        .set_global(robust_qconfig)
                        .set_object_type(torch.nn.Conv2d, robust_qconfig)
                        .set_object_type(torch.nn.Linear, robust_qconfig)
                        .set_object_type(torch.nn.ReLU, robust_qconfig)
                        .set_object_type(torch.nn.BatchNorm2d, robust_qconfig)
                    )

                    # try sample
                    example_inputs = (
                        torch.randn(1, 3, cfg.image_size[0], cfg.image_size[1]),
                    )
                    if device.type != "cpu":
                        work_model.cpu()  # FX usually requires CPU

                    prepared_model = prepare_fx(
                        work_model, qconfig_mapping, example_inputs
                    )

                    # Calibrate
                    with torch.no_grad():
                        for images, labels in calib_loader:
                            images = images.to("cpu")
                            prepared_model(images)

                    # Convert to INT8
                    q_model = convert_fx(prepared_model)

                elif method == "dynamic_int8":
                    # Note: This runs on CPU.
                    work_model.cpu()
                    q_model = torch.quantization.quantize_dynamic(
                        work_model,
                        {nn.Linear},  # Only quantize Linear layers dynamically usually
                        dtype=torch.qint8,
                    )

                elif method == "fp16":
                    # FP16 (Half Precision) - Requires GPU with half precision support.
                    # Simple casting.
                    work_model.eval()
                    if device.type == "cuda":
                        q_model = work_model.half()
                        q_model.to("cuda")
                    else:
                        # On CPU, FP16 inference might be slow or not supported for all ops.
                        q_model = work_model.half()

                # Measure Size
                q_size = print_size_of_model(q_model)
                logger.info(
                    f"{method} Size: {q_size:.2f} MB | Reduction: {fp32_size / q_size:.2f}x"
                )

                # Measure accuracy
                eval_device = (
                    "cuda" if (method == "fp16" and device.type == "cuda") else "cpu"
                )
                if eval_device == "cpu":
                    q_model.cpu()

                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(eval_device)
                        labels = labels.to(eval_device)

                        if method == "fp16":
                            images = images.half()

                        outputs = q_model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                acc = 100 * correct / total
                logger.info(f"{method} Accuracy: {acc:.2f}%")

                results.append(
                    {
                        "Fold": fold_id,
                        "Method": method,
                        "FP32 Size (MB)": fp32_size,
                        "Quant Size (MB)": q_size,
                        "Reduction": fp32_size / q_size,
                        "Accuracy": acc,
                    }
                )

                save_name = f"model_{method}.pth"
                save_path = os.path.join(fold_dir, save_name)
                torch.save(q_model.state_dict(), save_path)

            except Exception as e:
                logger.error(f"Quantization method {method} failed: {e}")
                import traceback

                traceback.print_exc()

    if results:
        # Prepare Summary df

        import pandas as pd
        from tabulate import tabulate

        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print(f"QUANTIZATION RESULTS ({cfg.model_type})")
        print("=" * 60)
        print(tabulate(df, headers="keys", tablefmt="grid"))

        avg_acc = df["Accuracy"].mean()
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")

        # Save Summary CSV
        df.to_csv(os.path.join(cfg.output_dir, "quantization_summary.csv"), index=False)


if __name__ == "__main__":
    main()
