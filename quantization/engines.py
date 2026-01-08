import torch
import torch.nn as nn
import torch.quantization
from torch.ao.quantization import quantize_fx
import time
import os
import copy
from tqdm import tqdm

class QuantizationEngine:
    def __init__(self, logger):
        self.logger = logger

    def measure_latency(self, model, input_dummy, num_runs=100):
        """
        Measures the average inference latency of the model.
        Running multiple times to get a stable estimate.
        """
        model.eval()
        # Auto-cast input to match model dtype (important for FP16)
        param_dtype = next(model.parameters()).dtype
        if input_dummy.dtype != param_dtype:
            input_dummy = input_dummy.to(dtype=param_dtype)

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_dummy)
            
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(input_dummy)
            end_time = time.time()
            
        avg_latency = (end_time - start_time) / num_runs * 1000 # ms
        return avg_latency

    def evaluate_accuracy(self, model, data_loader, device='cpu'):
        """
        Evaluates the accuracy of the model on the provided loader.
        """
        model.eval()
        correct = 0
        total = 0
        
        # Check model dtype
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating Accuracy"):
                images = images.to(device)
                labels = labels.to(device)
                
                # Cast Input if model is FP16
                if param_dtype == torch.float16:
                    images = images.half()
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def dynamic_quantize(self, model):
        """
        Applies Dynamic Quantization (Weights=INT8, Activations=INT8 dynamically).
        """
        self.logger.info("Applying Dynamic Quantization (INT8)...")
        # Need to deepcopy to avoid modifying baseline if passed as ref?
        # quantize_dynamic creates a new module structure.
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear},  # Only linear layers are dynamically quantized by default for floats
            dtype=torch.qint8
        )
        return quantized_model

    def dynamic_quantize_fp16(self, model):
        """
        Applies Native FP16 Cast.
        """
        self.logger.info("Casting model to FP16 (Native)...")
        import copy
        model_fp16 = copy.deepcopy(model)
        model_fp16.half()
        return model_fp16

    def static_quantize(self, model, calibration_loader, backend='fbgemm'):
        """
        Applies Static Quantization using FX Graph Mode (Recommended for ViT).
        """
        self.logger.info(f"Applying Static Quantization (FX Graph Mode - Backend: {backend})...")
        
        # 0. Configuration
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)
        
        # 1. Prepare (Symbolic Trace + Insert Observers)
        # We need a concrete input example for tracing
        example_inputs = next(iter(calibration_loader))[0]
        
        prepared_model = quantize_fx.prepare_fx(
            model, 
            qconfig_mapping, 
            example_inputs
        )
        self.logger.info("FX Graph prepared. Running Calibration...")
        
        # 2. Calibrate
        self._calibrate(prepared_model, calibration_loader)
        self.logger.info("Calibration Complete.")
        
        # 3. Convert
        quantized_model = quantize_fx.convert_fx(prepared_model)
        self.logger.info("Model Converted to INT8 (FX).")
        
        return quantized_model

    def _calibrate(self, model, loader):
        """
        Runs forward passes on the calibration data to gather statistics.
        """
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(loader, desc="Calibrating")):
                # Static Quantization requires CPU for standard PyTorch flow usually
                images = images.to('cpu') 
                _ = model(images)
