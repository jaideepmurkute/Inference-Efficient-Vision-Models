import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher_training.teacher_config import TeacherConfig
from teacher_training.utils import (
    get_logger,
    set_seed,
    get_dataloader,
    create_fold_split_idx,
    build_img_paths,
    save_checkpoint,
    load_checkpoint,
    visualize_training_log,
    create_model,
)
from teacher_training.train import train_one_epoch, validate

# ----------------------------------------------------------------------------


def main():
    cfg = TeacherConfig()
    logger = get_logger(cfg)
    logger.info(f"Configuration: {cfg}")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Build Data Paths & corresponding labels for easy access
    data_paths = build_img_paths(cfg)

    criterion = nn.CrossEntropyLoss()

    if cfg.choice == 1:
        img_paths = np.array(data_paths["train"]["img_paths"])
        cls_ids = np.array(data_paths["train"]["cls_ids"])

        fold_idx_dict = create_fold_split_idx(cfg, img_paths, cls_ids)
        save_checkpoint(cfg, fold_idx_dict=fold_idx_dict)

        results = []

        for fold_k in range(cfg.num_folds):
            logger.info(f"--- Starting Fold {fold_k} ---")

            train_idx = fold_idx_dict[fold_k]["train"]
            val_idx = fold_idx_dict[fold_k]["val"]

            train_loader = get_dataloader(
                cfg, "train", img_paths[train_idx], cls_ids[train_idx]
            )
            val_loader = get_dataloader(
                cfg, "val", img_paths[val_idx], cls_ids[val_idx]
            )

            logger.info(f"Creating model: {cfg.model_name}")
            model = create_model(cfg, device, logger)
            model.to(device)

            logger.info("Creating optimizer")
            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

            best_acc = 0.0
            training_log = {
                "train": {"loss": [], "accuracy": []},
                "validation": {"loss": [], "accuracy": []},
            }

            for epoch in range(cfg.epochs):
                train_loss, train_acc, epoch_time = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    logger,
                    DEBUG_MODE=cfg.DEBUG_MODE,
                )
                val_loss, val_acc = validate(
                    model, val_loader, criterion, device, DEBUG_MODE=cfg.DEBUG_MODE
                )

                train_acc *= 100
                val_acc *= 100

                training_log["train"]["loss"].append(train_loss)
                training_log["train"]["accuracy"].append(train_acc)
                training_log["validation"]["loss"].append(val_loss)
                training_log["validation"]["accuracy"].append(val_acc)

                logger.info(
                    f"Fold {fold_k} Epoch {epoch + 1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%"
                )

                if val_acc > best_acc:
                    # Save the best epoch checkpoint
                    best_acc = val_acc
                    save_checkpoint(
                        cfg,
                        model=model,
                        fold_id=fold_k,
                        suffix="best",
                        training_log=training_log,
                    )

            # Save latest epoch checkpoint
            save_checkpoint(
                cfg,
                model=model,
                fold_id=fold_k,
                suffix="last",
                training_log=training_log,
            )

            # Plot training history
            visualize_training_log(cfg, training_log, fold_id=fold_k)

            results.append({"fold": fold_k, "best_acc": best_acc})

        logger.info(f"Training Complete. Results: {results}")

    elif cfg.choice == 2:
        # Test Mode
        logger.info("Running Test on all folds...")
        test_img_paths = np.array(data_paths["test"]["img_paths"])
        test_cls_ids = np.array(data_paths["test"]["cls_ids"])
        test_loader = get_dataloader(cfg, "test", test_img_paths, test_cls_ids)

        results = []
        for fold_k in range(cfg.num_folds):
            ckpt_dir = os.path.join(cfg.output_dir, f"fold_{fold_k}")
            ckpt_path = os.path.join(ckpt_dir, f"model_{cfg.test_ckpt_type}.pth")

            if not os.path.exists(ckpt_path):
                logger.warning(f"Checkpoint not found for Fold {fold_k}: {ckpt_path}")
                continue

            logger.info(f"Eval Fold {fold_k} ({ckpt_path})...")

            original_pretrained = cfg.pretrained
            cfg.pretrained = False
            model = create_model(cfg, device, logger)
            cfg.pretrained = original_pretrained

            try:
                load_checkpoint(
                    cfg, "model", model=model, fold_id=fold_k, suffix=cfg.test_ckpt_type
                )
            except Exception as e:
                logger.error(f"Error loading checkpoint for fold {fold_k}: {e}")
                continue

            loss, acc = validate(
                model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
                DEBUG_MODE=cfg.DEBUG_MODE,
            )
            acc *= 100
            logger.info(f"Fold {fold_k} Test Acc: {acc:.2f}% | Loss: {loss:.4f}")
            results.append({"fold": fold_k, "test_acc": acc, "test_loss": loss})

        logger.info(f"Final Test Results: {results}")


if __name__ == "__main__":
    main()
