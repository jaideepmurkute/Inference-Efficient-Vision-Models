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
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating Accuracy"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def dynamic_quantize(self, model):
        """
        Applies Dynamic Quantization to the model.
        Quantizes Linear and LSTM layers to INT8.
        """
        self.logger.info("Applying Dynamic Quantization...")
        
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear},  # Only linear layers are dynamically quantized by default for floats
            dtype=torch.qint8
        )
        return quantized_model

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
