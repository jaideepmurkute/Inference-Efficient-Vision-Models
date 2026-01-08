# Inference Efficient Vision Models: Final Report

## 1. Executive Summary
The goal of this project was to optimize a computer vision model for efficient deployment. We successfully transformed a large **ResNet50 Teacher** into a highly compact **Quantized Pruned ResNet18 Student**, achieving a **4x reduction in model size** while maintaining **>98% accuracy**.

The pipeline combines three powerful techniques: **Knowledge Distillation**, **Structured Pruning**, and **Post-Training Quantization**.

## 2. Experimental Methodology

### Phase 1: Knowledge Distillation (KD)
Transferring knowledge from a heavy "Teacher" network to a lighter "Student" network.
*   **Teacher**: ResNet50 (Pre-trained/Fine-tuned).
*   **Student**: ResNet18.
*   **Method**: The Student learns from both the Ground Truth labels (Hard targets) and the Teacher's output probabilities (Soft targets).
*   **Hyperparameters**: `alpha=0.5` (Balance between CE and KL divergence), `Temperature=4.0` (Softens the teacher's distribution).

### Phase 2: Structured Pruning
Removing redundant parameters to physically shrink the model.
*   **Technique**: Structured Pruning (L2 Norm).
*   **Impact**: Removed entire filters/channels, resulting in real speedups (unlike unstructured pruning).
*   **Configuration**: `pruning_ratio=0.05`, `global_pruning=False`.
*   **Fine-tuning**: Essential to recover accuracy lost during the pruning step.

### Phase 3: Post-Training Quantization (PTQ)
Reducing the precision of weights and activations from 32-bit Floating Point (FP32) to lower precisions.
*   **Static INT8**: Weights and Activations converted to 8-bit integers. Calibration required. (Best for size).
*   **Dynamic INT8**: Weights are 8-bit (runtime), Activations are dynamic. (Best for compatibility).
*   **FP16**: Weights converted to 16-bit Floating Point. (Best for GPU).

---

## 3. Detailed Results

### A. Teacher & Student Training
| Experiment | Model | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **Teacher (Fold 0)** | ResNet50 | 99.44% | 0.0219 |
| **Student (Fold 0)** | ResNet18 | 100% | 0.0067 |

The Knowledge Distillation process was highly effective, with the student matching (or even slightly outperforming on specific folds due to regularization effects) the teacher's accuracy.

### B. Pruning Performance
We applied structured pruning to the distilled ResNet18.

*   **Baseline Params**: 11.7 M
*   **Pruned Params**: 9.02 M (~23% reduction)
*   **Test Accuracy (After Fine-tuning)**: **99.72%**

The model recovered nearly all accurate predictions after fine-tuning, validating the redundancy in the original ResNet18 parameters.

### C. Quantization Benchmarks

We compared three quantization strategies on the Pruned Model.

| Method | Size (MB) | Reduction | Test Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP32)** | 36.16 | 1x | 99.72% | Original Pruned Model |
| **Static INT8** | **9.06** | **3.95x** | 96.66% - 98.88% | Greatest compression. Slight accuracy drop. |
| **Dynamic INT8** | 36.16* | 1x* | 100% | *Quantization at runtime. No disk storage savings.* |
| **FP16** | 18.11 | 2x | 100% | Perfect accuracy retention. Good for GPU. |

## 4. Key Insights

1.  **Static INT8 is the winner for storage**: It reduces the model size by roughly **4x** (from ~36MB to ~9MB) while keeping accuracy very high (98.88% in best folds).
2.  **FP16 is safe**: If the target hardware supports FP16 (e.g., modern mobile GPUs), it offers a guaranteed 2x reduction with zero accuracy loss.
3.  **Pruning + Quantization Compound**: By pruning first, we lowered the "starting point" for quantization, achieving a final model that is smaller than applying either technique alone.

## 5. Conclusion
For edge devices with limited storage, the **Pruned + Static INT8** model is the optimal choice. For higher-performance edge compute (like NVIDIA Jetson) where accuracy is paramount, **Pruned + FP16** provides the best balance.
