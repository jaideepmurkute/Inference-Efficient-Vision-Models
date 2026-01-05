
import time

import torch
from tqdm import tqdm

from .utils import calculate_accuracy

# ------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, logger):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    start_time = time.time()
    
    for i, (images, labels) in enumerate(tqdm(loader)):
        # if i == 2:
        #     break
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        pred = torch.argmax(outputs, dim=1)
        running_corrects += (pred == labels).sum().item()
        total_samples += images.size(0)
        
        # if (i + 1) % 10 == 0:
        #     logger.info(f"Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_time


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            # if i == 2:
            #     break
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            pred = torch.argmax(outputs, dim=1)
            running_corrects += (pred == labels).sum().item()
            total_samples += images.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc


def test(model, loader, device, logger):
    model.eval()
    running_corrects = 0
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            # if i == 2:
            #     break
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            running_corrects += (pred == labels).sum().item()
            total_samples += images.size(0)
            
    acc = running_corrects / total_samples
    inference_time = (time.time() - start_time) / total_samples * 1000 # ms per sample roughly
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Inference Time per Batch/Sample proxy: {inference_time:.4f} ms") # Rough proxy
    
    return acc
