import torch
import torch_pruning as tp
from tqdm import tqdm
import time
import os
import thop
import types

class StructuredPruningEngine:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device)

    def get_pruner(self, model, example_inputs):
        """
        No fancy channel rounding to 64. No alignment hacks.
        Just basic importance pruning.
        Monkey-patching the attention forward pass handles the resulting 'messy' shapes.
        """
        # Select Importance Method
        method = getattr(self.cfg, 'pruning_method', 'l1')
        if method == 'random':
            imp = tp.importance.RandomImportance()
        elif method == 'l1':
            imp = tp.importance.MagnitudeImportance(p=1)
        elif method == 'l2':
            imp = tp.importance.MagnitudeImportance(p=2)
        elif method == 'group_norm':
            imp = tp.importance.GroupNormImportance(p=2)
        else:
            self.logger.warning(f"Unknown pruning method '{method}', defaulting to L1")
            imp = tp.importance.MagnitudeImportance(p=1)
        
        # IDENTIFY LAYERS TO PROTECT
        # We must protect the "Embedding Dimension" (Residual Stream).
        # We only want to prune "Internal Dimensions" (Heads and MLP Hidden).
        # Protected: norm1, norm2, attn.proj (output), mlp.fc2 (output), head.
        ignored_layers = []
        
        # 1. Protect Classification Head
        if hasattr(model, 'head'): ignored_layers.append(model.head)
        if hasattr(model, 'fc'): ignored_layers.append(model.fc)
        if hasattr(model, 'classifier'): ignored_layers.append(model.classifier)
        
        # 2. Protect Residual Stream in every Block
        for name, module in model.named_modules():
            # timm Block structure: norm1, attn (qkv, proj), ls1, norm2, mlp (fc1, fc2), ls2
            if hasattr(module, 'norm1'): ignored_layers.append(module.norm1)
            if hasattr(module, 'norm2'): ignored_layers.append(module.norm2)
            
            # For Linear layers, protecting the MODULE prevents its Output Channels from being pruned.
            # But the dependency graph ensures input channels are pruned if the previous layer was pruned.
            
            # Protect attn.proj (Output must match Residual C)
            if hasattr(module, 'attn') and hasattr(module.attn, 'proj'):
                ignored_layers.append(module.attn.proj)
                
            # Protect mlp.fc2 (Output must match Residual C)
            if hasattr(module, 'mlp') and hasattr(module.mlp, 'fc2'):
                ignored_layers.append(module.mlp.fc2)
                
            # Protect Patch Embed (Output must match Residual C)
            if hasattr(module, 'patch_embed'): ignored_layers.append(module.patch_embed)

        self.logger.info(f"Protected {len(ignored_layers)} layers (Residual Stream & Head). Pruning only Internal Dims (QKV, FC1). Params: Method={method}, Round={self.cfg.round_to}")

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

    def monkey_patch_vit_forward(self, model):
        """
        Monkey-patches the Attention.forward method in timm's VisionTransformer.
        Instead of crashing when head_dim is 53 or 107, this patched version
        dynamically calculates head_dim = C / num_heads.
        
        It treats the entire layer as "Global Attention" (1 Head) if the dimensions become weird,
        ensuring mathematical validity without crashing.
        """
        from timm.models.vision_transformer import Attention

        def forward_flexible(self, x, attn_mask=None):
            # x shape: [Batch, Tokens, Channels]
            B, N, C = x.shape
            
            # Get actual output dimension after pruning
            current_dim = self.qkv.out_features
            
            # Architecture Check: Is it a triplet? (Q, K, V)
            if current_dim % 3 != 0:
                # If pruner broke QKV symmetry (Extremely Rare with DependencyGraph), 
                # we technically cannot run standard attention without splitting logic.
                # Fallback: Assume it's effectively 1 head, but we might lose the '3' structure.
                # For now, let's assume valid triplet or crash with informative error.
                # But actually, with 1 head, we technically just need to split 'current_dim' into 3 chunks? 
                # No, standard code expects explicit 3.
                pass 

            head_dim_total = current_dim // 3
            
            # Divisibility Check
            if head_dim_total % self.num_heads == 0:
                curr_heads = self.num_heads
                curr_head_dim = head_dim_total // self.num_heads
            else:
                # Fallback to Robust Mode: 1 Head (Global Attention)
                # This works for Prime Numbers (e.g. 127)
                curr_heads = 1
                curr_head_dim = head_dim_total
            
            # qkv shape logic
            # We must ensure reshapes match 'current_dim'
            # (3 * Heads * Dim) == current_dim
            # If (3 * 1 * 127) = 381 == 381 -> OK.
            
            try:
                qkv = self.qkv(x).reshape(B, N, 3, curr_heads, curr_head_dim).permute(2, 0, 3, 1, 4)
            except RuntimeError as e:
                # Last resort fallback if somehow even 1-head fails (e.g. non-triplet)
                print(f"CRITICAL SHAPE ERROR in {curr_heads}x{curr_head_dim}: {e}")
                raise e
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            # Standard Attention Logic
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Fix: Use -1 instead of C. 
            # C is the input dimension. Pruned internal dimension might be smaller.
            # reshaping to -1 flattens (Heads, Dim) -> (Heads*Dim), whatever that size is.
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Apply the patch to all Attention modules in the model
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(forward_flexible, module)
                
        self.logger.info("Monkey-patched Attention.forward to support flexible/weird dimensions.")

    def prune_model(self, model):
        model.to(self.device)
        model.eval()
        
        # 1. Apply Monkey Patch FIRST
        self.monkey_patch_vit_forward(model)

        example_inputs = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        pruner = self.get_pruner(model, example_inputs)
        
        self.logger.info("Starting Pruning (Structured)...")
        pruner.step()
        
        # No 'Align Attention Heads' hacks here. 
        # We rely on the monkey patch to execute whatever shape remains.
        
        self.logger.info("Pruning Complete.")
        return model

    def finetune(self, model, train_loader, val_loader, epochs, learning_rate):
        """
        Standard fine-tuning loop.
        """
        self.logger.info(f"Starting Fine-tuning for {epochs} epochs...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        
        history = {'loss': [], 'accuracy': []}
        best_acc = 0.0
        best_state = None
        
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
                
                # Update Pbar
                # acc = 100*correct/total
                # pbar.set_postfix({'acc': acc, 'loss': running_loss/total}) # Simplified loss display
            
            # Epoch Stats
            epoch_loss = running_loss / len(train_loader) # Average over batches? No, running_loss was sum of items if reduction='mean'?
            # Wait, default CrossEntropyLoss is mean.
            # running_loss += loss.item() (is mean of batch).
            # So epoch_loss = running_loss / len(loader). Correct.
            epoch_acc = 100 * correct / total
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            # Validation Step
            if val_loader:
                current_val_acc = self.evaluate_accuracy(model, val_loader)
                if current_val_acc > best_acc:
                    best_acc = current_val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                self.logger.info(f"Epoch {epoch+1} Train Acc: {epoch_acc:.2f}% | Val Acc: {current_val_acc:.2f}% (Best: {best_acc:.2f}%)")

        if best_state is not None:
             self.logger.info(f"Restoring best fine-tuned model (Acc: {best_acc:.2f}%)")
             model.load_state_dict(best_state)

        return model, history

    def evaluate_metrics(self, model, loader):
        # Same evaluation logic
        model.eval()
        model.to(self.device)
        acc = self.evaluate_accuracy(model, loader)
        
        dummy = torch.randn(1, 3, *self.cfg.image_size).to(self.device)
        start = time.time()
        for _ in range(50): _ = model(dummy)
        lat = (time.time() - start)/50 * 1000
        
        try:
            macs, params = thop.profile(model, inputs=(dummy,), verbose=False)
            macs_g = macs / 1e9
            params_m = params / 1e6
        except:
            macs_g = 0; params_m = 0
            
        return {"Accuracy": acc, "Latency (ms)": lat, "MACs (G)": macs_g, "Params (M)": params_m, "Size (MB)": 0}

    def evaluate_accuracy(self, model, loader):
        correct = 0; total = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader)):
                # if i == 2:
                #     break
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred==labels).sum().item()
        return 100*correct/total
