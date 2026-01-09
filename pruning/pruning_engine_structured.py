import torch
import torch_pruning as tp
from tqdm import tqdm
import time
import thop

# --------------------------------------------------------------------------


class StructuredPruningEngine:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)

    def get_pruner(self, model, example_inputs):
        # Select Importance Method
        method = getattr(self.cfg, "pruning_method", "l1")
        if method == "random":
            imp = tp.importance.RandomImportance()
        elif method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
        elif method == "l2":
            imp = tp.importance.MagnitudeImportance(p=2)
        elif method == "group_norm":
            # Assuming refers to L2 magnitude of the group.
            imp = tp.importance.MagnitudeImportance(p=2)
        elif method == "taylor":
            imp = tp.importance.GroupTaylorImportance()
        else:
            self.logger.warning(f"Unknown pruning method '{method}', defaulting to L1")
            imp = tp.importance.MagnitudeImportance(p=1)

        # Ignore layers we dont want to ever be pruned - input, & output classification head
        ignored_layers = []

        # 1. Protect Classification Head (Generic checks)
        if hasattr(model, "head"):
            ignored_layers.append(model.head)
        if hasattr(model, "fc"):
            ignored_layers.append(model.fc)
        if hasattr(model, "classifier"):
            ignored_layers.append(model.classifier)

        self.logger.info(
            f"Protected {len(ignored_layers)} layers (Head). Params: Method={method}, \
                Round={self.cfg.round_to}"
        )

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=self.cfg.pruning_ratio,
            round_to=self.cfg.round_to,
            global_pruning=False,
            ignored_layers=ignored_layers,
        )
        return pruner

    def prune_model(self, model):
        model.to(self.device)
        model.eval()

        # Standard input for dependency graph tracing
        example_inputs = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        pruner = self.get_pruner(model, example_inputs)

        self.logger.info("Starting Pruning (Structured)...")
        pruner.step()
        self.logger.info("Pruning Complete.")

        return model

    def finetune(self, model, train_loader, val_loader, epochs, learning_rate):
        """
        Simple finetuning loop.
        """
        self.logger.info(f"Starting Fine-tuning for {epochs} epochs...")
        # Use SGD for stability or AdamW with low LR
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        history = {"loss": [], "accuracy": []}
        best_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(tqdm(train_loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total

            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            if val_loader:
                current_val_acc = self.evaluate_accuracy(model, val_loader)
                if current_val_acc > best_acc:
                    best_acc = current_val_acc
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                self.logger.info(
                    f"Epoch {epoch + 1} Train Acc: {epoch_acc:.2f}% | \
                        Val Acc: {current_val_acc:.2f}% (Best: {best_acc:.2f}%)"
                )

        if best_state is not None:
            self.logger.info(f"Restoring best fine-tuned model (Acc: {best_acc:.2f}%)")
            model.load_state_dict(best_state)

        return model, history

    def evaluate_metrics(self, model, loader):
        model.eval()
        model.to(self.device)
        acc = self.evaluate_accuracy(model, loader)

        dummy = torch.randn(1, 3, *self.cfg.image_size).to(self.device)

        # Warmup for less noise from slow resource allocation
        for _ in range(10):
            _ = model(dummy)

        # Actual latency measurement
        start = time.time()
        for _ in range(50):
            _ = model(dummy)
        lat = (time.time() - start) / 50 * 1000

        try:
            macs, params = thop.profile(model, inputs=(dummy,), verbose=False)
            macs_g = macs / 1e9
            params_m = params / 1e6
        except Exception as e:
            macs_g = 0
            params_m = 0
            self.logger.warning(f"Failed to profile model: {e}")

        try:
            size = torch.save(model.state_dict(), "temp.pth")
            size_mb = size / 1e6
        except Exception as e:
            size_mb = 0
            self.logger.warning(f"Failed to measure model size: {e}")

        return {
            "Accuracy": acc,
            "Latency (ms)": lat,
            "MACs (G)": macs_g,
            "Params (M)": params_m,
            "Size (MB)": size_mb,
        }

    def evaluate_accuracy(self, model, loader):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        return 100 * correct / total
