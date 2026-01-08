import torch
import torch.nn as nn
import timm
import torchvision
import logging

def get_prunable_model(model_name, use_timm=True, **kwargs):
    """
    Returns a model (timm or torchvision).
    """
    if use_timm:
        model = timm.create_model(model_name, **kwargs)
    else:
        # Torchvision logic
        if not hasattr(torchvision.models, model_name):
             raise ValueError(f"Torchvision model {model_name} not found")
        
        # Parse kwargs for pretrained
        pretrained = kwargs.get('pretrained', False)
        weights = "DEFAULT" if pretrained else None
        
        model_fn = getattr(torchvision.models, model_name)
        model = model_fn(weights=weights)
        
        # Replace Head
        num_classes = kwargs.get('num_classes', 1000)
        
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
             if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
             elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
             else:
                 print(f"Warning: classifier head structure unknown for {model_name}")
    return model
