# Inference-Efficient Vision Models

This repository implements a multi-stage optimization pipeline for compact computer vision models. The goal is to minimize model size and latency for edge deployment while maintaining accuracy.

The pipeline optimizes from **ResNet50** (Teacher) to **ResNet18** (Student).

## Optimization Stages

1.  **Knowledge Distillation (KD)**: Compressing knowledge from the ResNet50 Teacher into the ResNet18 Student.
2.  **Structured Pruning**: Removing redundant channels/kernels to reduce parameters and FLOPs.
3.  **Post-Training Quantization (PTQ) & Casting**: Converting weights and activations to lower precision (INT8, FP16) for reduced memory footprint.

## Directory Structure

*   `teacher_training/`: Baseline training (ResNet50).
*   `knowledge_distillation/`: Distillation implementation.
*   `pruning/`: Structured pruning logic.
*   `quantization/`: Static and Dynamic quantization scripts.
*   `data/`: Dataset handling.
*   `REPORT.md`: Detailed experiment logs and analysis.

## Usage

**1. Setup**
```bash
pip install -r requirements.txt
```

**2. Running the Pipeline**
The stages are designed to be run sequentially.

**Stage 1: Teacher Training**
```bash
python teacher_training/main.py
```

**Stage 2: Knowledge Distillation**
```bash
python knowledge_distillation/main.py
```

**Stage 3: Pruning**
```bash
python pruning/main.py
```

**Stage 4: Quantization**
```bash
python quantization/main.py
```

## Results

Performance comparison across the optimization pipeline.

| Model Stage | Model | Params (M) | Size (MB) | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher** | ResNet50 | ~25.5 | ~100 | ~99.9% | Baseline |
| **Distilled Student** | ResNet18 | 11.7 | ~45 | 100% |  |
| **Distilled + Pruned Student** | ResNet18 | 9.02 | ~36 | 99.72% | ~20% Sparsity |
| **Distilled + Pruned + Quantized Student (FP16 Casting)** | ResNet18 | 9.02 | 18.11 | 99.72% | |
| **Distilled + Pruned + Quantized Student (INT8)**| ResNet18 | 9.02 | **9.06** | 98.88% | **>10x Compression vs Teacher** |

*See `REPORT.md` for detailed metrics per fold.*

## License
Educational/Research use.
