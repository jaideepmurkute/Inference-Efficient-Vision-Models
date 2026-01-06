import torch
import torch.nn as nn
import timm

def get_prunable_model(model_name, **kwargs):
    """
    Returns a standard timm model. 
    Monkey-patching logic is now handled dynamically in pruning_engine_structured.py 
    so we don't need a custom class definition here that might conflict.
    """
    model = timm.create_model(model_name, **kwargs)
    return model
