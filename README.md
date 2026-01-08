# Inference Efficient Vision Models

## Overview
This project demonstrates a complete pipeline for creating highly efficient computer vision models suitable for edge deployment. It implements a multi-stage optimization strategy applied to Vision Transformers (ViT) on the **NEU-DET** steel surface defect dataset.

**Key Techniques:**
1.  **Teacher-Student Knowledge Distillation**: Compressing a ViT-Base into a ViT-Tiny.
2.  **Structured Pruning**: Physically removing 20% of channels/heads from the student model.
3.  **Post-Training Quantization**: Compressing weights to INT8 and FP16 for reduced memory usage.

## 📂 Directory Structure

*   `teacher_training/`: Scripts to train the large baseline model (Teacher).
*   `knowledge_distillation/`: Distills knowledge from Teacher to Student.
*   `pruning/`: Applies structured pruning to the distilled student.
*   `quantization/`: Applies Dynamic/Static quantization to the pruned model.
*   `data/`: Dataset storage.
*   `REPORT.md`: Detailed analysis of experimental results and performance metrics.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Step 1: Train Teacher Model**
Establish a high-accuracy baseline.
```bash
python teacher_training/main.py
```

**Step 2: Knowledge Distillation**
Train a lightweight "Student" model using the Teacher's guidance.
```bash
python knowledge_distillation/main.py
```

**Step 3: Structured Pruning**
Physically remove redundant channels from the Student model and fine-tune to recover accuracy.
*   *Output*: Saves a full model object (e.g., `pruned_model_r0.2.pth`).
```bash
python pruning/main.py
```

**Step 4: Quantization**
Apply INT8/FP16 compression to the Pruned model and benchmark latency/size/accuracy.
```bash
python quantization/main.py
```

## 📊 Results Snapshot

| Model Stage | Size (MB) | Accuracy | Latency (CPU) |
| :--- | :--- | :--- | :--- |
| **Original Student** | ~22 MB | ~40% | ~15 ms |
| **Pruned (r=0.2)** | **17.6 MB** | **45%** | **~61 ms** |
| **Quantized (INT8)**| **5.2 MB** | **44%** | **~64 ms** |

> **Note**: While FP16 Quantization reduces size to **8.9 MB**, it incurs high latency on CPUs due to lack of native hardware support. See `REPORT.md` for full details.

## 📝 License
This project is for educational and research purposes.
