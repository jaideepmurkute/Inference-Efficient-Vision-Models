import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import time
import os
import thop

class UnstructuredPruningEngine:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)

    def prune_model(self, model):
        """
        Applies unstructured L1 pruning to all Linear and Conv2d layers.
        This simply sets weights to zero (masking) without changing tensor shapes.
        """
        self.logger.info(f"Starting Unstructured Pruning (Ratio: {self.cfg.pruning_ratio})...")
        model.to(self.device)
        
        prunable_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prunable_modules.append((module, 'weight'))
        
        # Apply global unstructured pruning
        # This determines the threshold globally (across all layers) or locally depending on user preference.
        # But 'prune.global_unstructured' is simpler and usually better.
        
        if self.cfg.global_pruning:
             prune.global_unstructured(
                prunable_modules,
                pruning_method=prune.L1Unstructured,
                amount=self.cfg.pruning_ratio,
            )
        else:
            # Local pruning
            for module, name in prunable_modules:
                prune.l1_unstructured(module, name=name, amount=self.cfg.pruning_ratio)

        # Make pruning permanent (remove mask buffers, integrate into weights)
        for module, name in prunable_modules:
            prune.remove(module, name)

        self.logger.info("Unstructured Pruning Complete. (Note: File size/FLOPs may not decrease without sparse support)")
        return model

    def finetune(self, model, train_loader, val_loader, epochs, learning_rate):
        """
        Simple fine-tuning loop.
        """
        self.logger.info(f"Starting Fine-tuning for {epochs} epochs with LR={learning_rate}...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model.to(self.device)
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                # if i == 2:
                #     break
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
                
                pbar.set_postfix({'loss': running_loss/total, 'acc': 100*correct/total})
            
            val_acc = self.evaluate_accuracy(model, val_loader)
            self.logger.info(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
            
        return model

    def evaluate_metrics(self, model, loader):
        """
        Calculates stats of the pruned model.
        Note: For unstructured pruning, standard MACs/Params tools might report original size
        unless they support zero-counting. We will manually count non-zero params.
        """
        model.eval()
        model.to(self.device)
        
        acc = self.evaluate_accuracy(model, loader)
        
        dummy_input = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        
        # Latency
        for _ in range(10): _ = model(dummy_input)
        start_time = time.time()
        for _ in range(100): _ = model(dummy_input)
        latency_ms = (time.time() - start_time) / 100 * 1000
        
        # MACs/Params (THOP generic)
        try:
            macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
            macs_g = macs / 1e9
        except:
            macs_g = 0.0 # Fallback

        # Check actual sparsity
        total_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        self.logger.info(f"Actual Model Sparsity: {sparsity*100:.2f}%")

        params_m = (total_params - zero_params) / 1e6 # Effective params
        
        # Size on disk
        torch.save(model.state_dict(), "temp_prune.p")
        size_mb = os.path.getsize("temp_prune.p") / 1e6
        os.remove('temp_prune.p')
        
        return {
            "Accuracy": acc,
            "Latency (ms)": latency_ms,
            "MACs (G)": macs_g,
            "Params (M)": params_m, # Reporting effective non-zero params
            "Size (MB)": size_mb
        }

    def evaluate_accuracy(self, model, loader):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader)):
                # if i == 2:
                #     break
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
