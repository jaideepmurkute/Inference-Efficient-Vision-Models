import math
import os
import time

import torch
import torch_pruning as tp
from tqdm import tqdm
import thop


import torch.nn.functional as F
import timm.models.vision_transformer


class PruningEngine:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)
        
    def get_pruner(self, model, example_inputs):
        """
        Initializes the dependency graph and pruner.
        """
        self.logger.info(f"Initializing Pruner (Ratio: {self.cfg.pruning_ratio}, Method: {self.cfg.pruning_method})...")
        
        # Importance Method
        if self.cfg.pruning_method == 'l1':
            importance = tp.importance.MagnitudeImportance(p=1)
        elif self.cfg.pruning_method == 'random':
            importance = tp.importance.RandomImportance()
        else:
            raise ValueError(f"Unknown pruning method: {self.cfg.pruning_method}")
        
        # Pruner
        # 'global_pruning=True' means we compare importance across ALL layers and prune the absolute
        # least important 20% of channels in the entire model. 
        # 'global_pruning=False' would prune 20% of channels in *each* layer individually.
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=importance,
            global_pruning=self.cfg.global_pruning,
            ch_sparsity=self.cfg.pruning_ratio, # Target sparsity
            round_to=self.cfg.round_to, # Align channels for hardware efficiency
            ignored_layers=[], # Can add classifier head here if we want to keep it fixed
        )
        return pruner

    def prune_model(self, model):
        """
        Performs the pruning step.
        """
        model.to(self.device)
        model.eval()
        
        # Dummy input for dependency graph
        example_inputs = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        
        pruner = self.get_pruner(model, example_inputs)
        
        self.logger.info("Starting Pruning...")
        
        # Torch-Pruning step() handles the magic
        pruner.step()
        
        # --- FIX FOR TIMM ViT ---
        # Torch-Pruning prunes the weights but doesn't update the 'num_heads' attribute in timm Attention.
        # We must manually fix this to avoid "RuntimeError: shape invalid" in forward().
        for name, module in model.named_modules():
            if "Attention" in str(type(module)):
                if hasattr(module, "num_heads") and hasattr(module, "qkv") and hasattr(module, "head_dim"):
                    
                    # Check if qkv was pruned
                        # Fix/Align Attention Heads
                        self._align_attention_heads(module, name)
                            
        self.logger.info("Pruning Complete.")
        return model

    def _align_attention_heads(self, module, name):
        """
        Ensures that the pruned QKV layer has dimensions compatible with the multi-head attention structure.
        
        CRITICAL FIX: We must NOT change 'head_dim' (e.g. 64 -> 53) to fit the pruned count.
        Doing so "squashes" different heads together, mixing features and destroying accuracy (16%).
        We must STRICTLY preserve 'head_dim' and only change 'num_heads'.
        
        Strategy:
        1. Keep head_dim = 64.
        2. Round 'total_dim' to the nearest multiple of 64.
        3. Pad with zeros (if growing) or Truncate (if shrinking) to match.
        """
        if not hasattr(module, "head_dim") or not hasattr(module, "qkv"):
            return
            
        current_out = module.qkv.out_features
        
        # 1. Get Original Head Dim (e.g. 64)
        original_head_dim = getattr(module, "head_dim", 64)
        if original_head_dim <= 0: original_head_dim = 64
        
        total_dim = current_out // 3
        
        # 2. Calculate Target Configuration
        # strictly preserve head_dim
        new_head_dim = original_head_dim
        
        # Strategy: ALWAYS PAD, NEVER TRUNCATE.
        # If the pruner kept 94 channels (1.4 heads), it means those 94 are important.
        # If we round down to 1 head (64), we delete 30 important channels -> Accuracy Crash.
        # If we round up to 2 heads (128), we pad 34 zeros. This preserves all info.
        
        # We uses ceil() to ensure we cover all surviving channels.
        new_heads = max(1, math.ceil(total_dim / original_head_dim))
        
        # Target aligned dimension
        aligned_total_dim = new_heads * new_head_dim
        
        # 3. Resize Weights (Pad or Truncate)
        if aligned_total_dim != total_dim:
            # logic should always be Padding now (since aligned >= total), but let's keep generic check
            action = "Padding" if aligned_total_dim >= total_dim else "Truncating"
            self.logger.info(f"Aligning {name}: {action} {total_dim}->{aligned_total_dim} to match heads={new_heads}x{new_head_dim}")
            
            # Reconstruct the fused tensor size
            new_out = 3 * aligned_total_dim
            
            old_weight = module.qkv.weight.data
            old_bias = module.qkv.bias.data if module.qkv.bias is not None else None
            
            new_weight = torch.zeros((new_out, module.qkv.in_features), device=self.device)
            new_bias = torch.zeros((new_out), device=self.device) if old_bias is not None else None
                
            # Copy Q, K, V blocks
            for i in range(3): 
                old_start = i * total_dim
                new_start = i * aligned_total_dim
                
                # How much valid data do we have?
                valid_len = min(total_dim, aligned_total_dim)
                
                # Copy valid data
                new_weight[new_start : new_start+valid_len] = old_weight[old_start : old_start+valid_len]
                if new_bias is not None:
                    new_bias[new_start : new_start+valid_len] = old_bias[old_start : old_start+valid_len]
                    
                # If padding, the rest stays 0.0 (safe for attention)
            
            # Apply changes physically to the layer
            module.qkv.weight = torch.nn.Parameter(new_weight)
            if new_bias is not None:
                module.qkv.bias = torch.nn.Parameter(new_bias)
            module.qkv.out_features = new_out
            
            # 2.b ALIGN PROJECTION LAYER (if present)
            if hasattr(module, 'proj') and isinstance(module.proj, torch.nn.Linear):
                proj = module.proj
                if proj.in_features == total_dim: 
                     self.logger.info(f"Aligning {name}.proj: {action} input {total_dim}->{aligned_total_dim}")
                     
                     old_proj_weight = proj.weight.data
                     
                     # New Weight (Out, New_In)
                     new_proj_weight = torch.zeros((proj.out_features, aligned_total_dim), device=self.device)
                     
                     # Copy valid columns
                     valid_len = min(total_dim, aligned_total_dim)
                     new_proj_weight[:, :valid_len] = old_proj_weight[:, :valid_len]
                     
                     proj.weight = torch.nn.Parameter(new_proj_weight)
                     proj.in_features = aligned_total_dim

        # 3. Update Module Metadata
        if new_heads != module.num_heads or new_head_dim != module.head_dim:
            self.logger.info(f"Updated {name}: heads={module.num_heads}->{new_heads}, dim={module.head_dim}->{new_head_dim}")
            module.num_heads = new_heads
            module.head_dim = new_head_dim
            module.scale = new_head_dim ** -0.5

    def finetune(self, model, train_loader, val_loader, epochs, learning_rate):
        """
        Simple fine-tuning loop to recover accuracy.
        """
        self.logger.info(f"Starting Fine-tuning for {epochs} epochs with LR={learning_rate}...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
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
            
            # Validation after epoch
            val_acc = self.evaluate_accuracy(model, val_loader)
            self.logger.info(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
            
        return model

    def evaluate_metrics(self, model, loader):
        """
        Calculates Accuracy, Latency, MACs, and Params.
        """
        model.eval()
        model.to(self.device)
        
        # 1. Accuracy
        acc = self.evaluate_accuracy(model, loader)
        
        # 2. Latency (CPU or GPU based on config, but usually we care about deployment latency)
        # Using self.device for now
        dummy_input = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        
        # Warmup
        for _ in range(10): _ = model(dummy_input)
        
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        latency_ms = (time.time() - start_time) / 100 * 1000
        
        # 3. MACs (Multiply-Accumulate Operations) & Params using THOP
        # THOP expects input on the same device
        macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        macs_g = macs / 1e9
        params_m = params / 1e6
        
        # 4. Model Size (MB)
        torch.save(model.state_dict(), "temp_prune.p")
        size_mb = os.path.getsize("temp_prune.p") / 1e6
        os.remove('temp_prune.p')
        
        return {
            "Accuracy": acc,
            "Latency (ms)": latency_ms,
            "MACs (G)": macs_g,
            "Params (M)": params_m,
            "Size (MB)": size_mb
        }
    
    def evaluate_accuracy(self, model, loader):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
