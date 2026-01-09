import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

# ---------------------------------------------------------------------------


def train_kd_one_epoch(
    student_model,
    teacher_model,
    loader,
    optimizer,
    criterion_ce,
    criterion_kd,
    alpha,
    temperature,
    device,
    logger,
    DEBUG_MODE,
):
    student_model.train()
    teacher_model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    start_time = time.time()

    for i, (images, labels) in enumerate(tqdm(loader)):
        if DEBUG_MODE and i == 2:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Student forward
        student_logits = student_model(images)

        # Teacher forward
        with torch.no_grad():
            teacher_logits = teacher_model(images)

        # Classification loss
        loss_ce = criterion_ce(student_logits, labels)

        # Divergence loss - KL-Divergence between student and teacher logits
        # logits scaled by temperature
        loss_kd = criterion_kd(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
        ) * (temperature * temperature)

        # Total Loss
        loss = (1.0 - alpha) * loss_ce + alpha * loss_kd

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = torch.argmax(student_logits, dim=1)
        running_corrects += (pred == labels).sum().item()
        total_samples += images.size(0)

        # if (i + 1) % 10 == 0:
        #     logger.info(f"Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f})")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_time


def validate(model, loader, criterion, device, DEBUG_MODE):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            if DEBUG_MODE and i == 2:
                break
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


def test(model, loader, device, logger, DEBUG_MODE):
    model.eval()
    running_corrects = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            if DEBUG_MODE and i == 2:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            running_corrects += (pred == labels).sum().item()
            total_samples += images.size(0)

    acc = running_corrects / total_samples
    inference_time = (time.time() - start_time) / total_samples * 1000  # ms per sample

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Inference time per batch: {inference_time:.4f} ms")

    return acc
