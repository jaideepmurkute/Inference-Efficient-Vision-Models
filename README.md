# Inference Efficient Vision Models

## Overview
This project enables the creation of highly efficient computer vision models optimized for edge deployment. It implements a comprehensive pipeline that transforms heavy, high-accuracy models into lightweight, fast-inference versions without significant accuracy loss.

We utilize a multi-stage optimization strategy applied to **ResNet** architectures (Teacher: ResNet50, Student: ResNet18).

**Key Techniques:**
1.  **Knowledge Distillation (KD)**: Compressing the knowledge of a ResNet50 Teacher into a compact ResNet18 Student.
2.  **Structured Pruning**: Removing redundant channels and kernels from the student model to reduce parameter count and FLOPs.
3.  **Post-Training Quantization (PTQ)**: Converting model weights and activations to lower precision (INT8, FP16) for reduced memory footprint and improved latency.

## 📂 Directory Structure

*   `teacher_training/`: Training scripts for the high-accuracy baseline (Teacher).
*   `knowledge_distillation/`: Scripts to distill the Teacher's knowledge into the Student.
*   `pruning/`: Structured pruning implementation to sparsify the Student model.
*   `quantization/`: Static and Dynamic quantization experiments (INT8, FP16).
*   `data/`: Directory for dataset storage.
*   `REPORT.md`: Detailed analysis of experimental results and methodology.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

The pipeline is modular. You can run each stage independently, provided the previous stage's output is available.

**Step 1: Teacher Training**
Train a ResNet50 to establish a high-accuracy baseline.
```bash
python teacher_training/main.py
```

**Step 2: Knowledge Distillation**
Train a ResNet18 Student using the Teacher's guidance (Soft targets) + Hard targets.
```bash
python knowledge_distillation/main.py
```

**Step 3: Structured Pruning**
Physically remove least important channels from the Student (ResNet18) and fine-tune.
```bash
python pruning/main.py
```

**Step 4: Quantization**
Apply INT8/FP16 quantization to the Pruned model to minimize size.
```bash
python quantization/main.py
```

## 📊 Results Summary

Our experiments demonstrate massive efficiency gains while maintaining nearly perfect accuracy.

| Model Stage | Model | Params (M) | Size (MB) | Accuracy (Test) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher** | ResNet50 | ~25.5 | ~100 | **~99.9%** | Baseline |
| **Student (Distilled)** | ResNet18 | 11.7 | ~45 | **100%** | Knowledge Transfer Success |
| **Pruned** | ResNet18 | 9.02 | ~36 | **99.72%** | 20% Pruning Ratio |
| **Quantized (Static INT8)**| ResNet18 | 9.02 | **9.06** | **98.88%** | **4x Size Reduction** |
| **Quantized (FP16)** | ResNet18 | 9.02 | **18.11** | **99.72%** | 2x Size Reduction |

*See `REPORT.md` for a deeper breakdown of each fold, configuration parameters (alpha, temperature), and specific quantization techniques.*

## 📝 License
This project is for educational and research purposes.
