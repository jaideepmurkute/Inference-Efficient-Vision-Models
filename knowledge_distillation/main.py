import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_distillation.kd_config import KDConfig
from knowledge_distillation.utils import (
    create_model,
    get_logger,
    set_seed,
    build_img_paths,
    create_fold_split_idx,
    save_checkpoint,
    get_dataloader,
    visualize_training_log,
)
from knowledge_distillation.train import train_kd_one_epoch, validate

# -------------------------------------------------------------------------


def create_and_load_teacher_model(cfg, teacher_output_root, device, fold_id, logger):
    teacher_model = create_model(
        cfg.teacher_model,
        cfg.use_timm,
        pretrained=False,
        num_classes=cfg.num_classes,
        logger=logger,
    )
    teacher_model.to(device)
    teacher_model.eval()

    teacher_ckpt_path = os.path.join(
        teacher_output_root, f"fold_{fold_id}", "model_best.pth"
    )

    if os.path.exists(teacher_ckpt_path):
        logger.info(f"Loading teacher weights from: {teacher_ckpt_path}")
        state = torch.load(teacher_ckpt_path, map_location=device)

        # Check for DataParallel 'module.' prefix just in case
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        teacher_model.load_state_dict(new_state)
        return teacher_model
    else:
        logger.error(
            f"Teacher checkpoint not found at {teacher_ckpt_path}. Cannot perform KD without teacher."
        )
        return None


def main():
    cfg = KDConfig()

    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    data_paths = build_img_paths(cfg)

    # Teacher experiment/ckpt path (relative location)
    teacher_output_root = os.path.join(
        "..", "teacher_training", "output", cfg.teacher_exp_name
    )

    if cfg.choice == 1:
        img_paths = np.array(data_paths["train"]["img_paths"])
        cls_ids = np.array(data_paths["train"]["cls_ids"])

        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)

        for fold_id in range(cfg.num_folds):
            logger.info(f"Training Fold: {fold_id}")

            train_loader = get_dataloader(
                cfg,
                split_type="train",
                img_paths=img_paths[fold_idx_dict[fold_id]["train"]],
                cls_ids=cls_ids[fold_idx_dict[fold_id]["train"]],
            )

            val_loader = get_dataloader(
                cfg,
                split_type="validation",
                img_paths=img_paths[fold_idx_dict[fold_id]["validation"]],
                cls_ids=cls_ids[fold_idx_dict[fold_id]["validation"]],
            )

            logger.info(
                f"Loading Teacher Model: {cfg.teacher_model} for Fold {fold_id}"
            )
            # ----------------------
            # Create & Load Teacher Model
            teacher_model = create_and_load_teacher_model(
                cfg, teacher_output_root, device, fold_id, logger
            )
            if teacher_model is None:
                raise ValueError("Teacher model not loaded. Cannot perform KD.")
            teacher_model.to(device)
            teacher_model.eval()
            # ----------------------
            # Create Student Model
            logger.info(f"Creating Student Model: {cfg.student_model}")
            student_model = create_model(
                cfg.student_model,
                cfg.use_timm,
                pretrained=True,
                num_classes=cfg.num_classes,
                logger=logger,
            )
            student_model.to(device)
            # ----------------------
            criterion_ce = nn.CrossEntropyLoss()
            criterion_kd = nn.KLDivLoss(reduction="batchmean")
            optimizer = optim.AdamW(student_model.parameters(), lr=cfg.learning_rate)
            # ----------------------
            best_acc = 0.0
            training_log = {
                "train": {"loss": [], "accuracy": []},
                "validation": {"loss": [], "accuracy": []},
                "epoch_times": [],
            }

            for epoch in range(cfg.epochs):
                train_loss, train_acc, epoch_time = train_kd_one_epoch(
                    student_model,
                    teacher_model,
                    train_loader,
                    optimizer,
                    criterion_ce,
                    criterion_kd,
                    cfg.alpha,
                    cfg.temperature,
                    device,
                    logger,
                    cfg.DEBUG_MODE,
                )
                val_loss, val_acc = validate(
                    student_model, val_loader, criterion_ce, device, cfg.DEBUG_MODE
                )

                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
                )

                training_log["train"]["loss"].append(train_loss)
                training_log["train"]["accuracy"].append(train_acc)
                training_log["validation"]["loss"].append(val_loss)
                training_log["validation"]["accuracy"].append(val_acc)
                training_log["epoch_times"].append(epoch_time)

                if val_acc > best_acc:
                    # Save the best checkpoint
                    best_acc = val_acc
                    save_checkpoint(
                        cfg,
                        model=student_model,
                        fold_id=fold_id,
                        suffix="best",
                        training_log=training_log,
                    )
                    logger.info(f"Saved Best Student Model (Acc: {best_acc:.2f}%)")

            # Save last checkpoint
            save_checkpoint(
                cfg,
                model=student_model,
                fold_id=fold_id,
                suffix="last",
                training_log=training_log,
            )

            # Plot training log
            visualize_training_log(cfg, training_log, fold_id=fold_id)

    elif cfg.choice == 2:
        logger.info("Running Test on all folds...")

        test_img_paths = np.array(data_paths["test"]["img_paths"])
        test_cls_ids = np.array(data_paths["test"]["cls_ids"])
        test_loader = get_dataloader(
            cfg, split_type="test", img_paths=test_img_paths, cls_ids=test_cls_ids
        )

        results = []
        use_timm_flag = getattr(cfg, "use_timm", True)

        for fold_id in range(cfg.num_folds):
            ckpt_path = os.path.join(
                cfg.output_dir, f"fold_{fold_id}", "model_best.pth"
            )
            if not os.path.exists(ckpt_path):
                continue

            logger.info(f"Testing Fold: {fold_id} ...")
            student_model = create_model(
                cfg.student_model,
                use_timm_flag,
                pretrained=False,
                num_classes=cfg.num_classes,
                logger=logger,
            )
            student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            student_model.to(device)
            # ----------------------
            loss, acc = validate(
                student_model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
                cfg.DEBUG_MODE,
            )
            logger.info(f"Fold {fold_id} Test Acc: {acc:.2f}% | Loss: {loss:.4f}")
            results.append({"fold": fold_id, "acc": acc, "loss": loss})

        logger.info(f"Final Test Results: {results}")


if __name__ == "__main__":
    main()
