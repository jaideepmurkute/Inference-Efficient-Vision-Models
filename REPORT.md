# Optimization Report: Inference-Efficient Vision Models

This report details the technical methodology and results of optimizing a vision model pipeline for edge deployment. The process involves Knowledge Distillation (KD), Structured Pruning, and Post-Training Quantization (PTQ), transitioning from a ResNet50 baseline to a compressed ResNet18.

## 1. Methodology

The optimization pipeline consists of three sequential stages designed to reduce model complexity while preserving feature extraction capabilities.

### 1.1 Knowledge Distillation (KD)
We compress the representation capability of a larger **ResNet50 (Teacher)** into a smaller **ResNet18 (Student)**.
*   **Loss Function**: Combination of Kullback-Leibler (KL) Divergence for soft targets and Cross-Entropy for hard targets.
*   **Key Parameters**:
    *   `alpha=0.5`: Balances the importance of teacher guidance vs. ground truth.
    *   `temperature=4.0`: Smooths the teacher's probability distribution to expose class relationships.

### 1.2 Structured Pruning
Post-distillation, we remove redundant kernels from the ResNet18 Student to reduce computational inference cost (FLOPs).
*   **Criterion**: L2 Norm ranking. Filters with the lowest magnitude are pruned.
*   **Strategy**: Global structured pruning (removing entire filters).
*   **Sparsity**: ~23% reduction in parameters.
*   **Recovery**: Fine-tuning is performed post-pruning to realign weights.

### 1.3 Post-Training Quantization & Precision Reduction
The final stage reduces memory precision.
*   **Static INT8**: 8-bit integer weights/activations. Requires forward-pass calibration to determine quantization ranges.
*   **Dynamic INT8**: 8-bit weights, dynamic activation quantization at runtime.
*   **FP16**: 16-bit floating point reduction (half-precision). (Technically casting)

---

## 2. Experimental Results

Experiments were conducted using cross-validation. Metrics reported below represent performance on the test set.

### 2.1 Baseline & Distillation Performance
Comparison of the heavy teacher model vs. the uncompressed student model trained via distillation.

| Model | Architecture | Params (M) | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | ResNet50 | ~25.5 | 99.44% | 0.0219 |
| **Student** | ResNet18 | 11.7 | 100% | 0.0067 |

*Note: The student achieves parity with the teacher, confirming effective knowledge transfer.*

### 2.2 Pruning Effectiveness
Structured pruning applied to the distilled ResNet18.

| Metric | Pre-Pruning | Post-Pruning (Fine-Tuned) | Delta |
| :--- | :--- | :--- | :--- |
| **Parameters** | 11.7 M | 9.02 M | -2.68 M (~23%) |
| **Accuracy** | 100% | 99.72% | -0.28% |
| **Model Size** | ~45 MB | ~36.16 MB | -20% |

### 2.3 Quantization & Precision Reduction Comparison
Quantization applied to the **Pruned** ResNet18 model.

| Quantization Mode | Physical Size (MB) | Compression Factor | Accuracy | Trade-off Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Baseline)** | 36.16 | 1.0x | 99.72% | Reference point. |
| **FP16 (Casting)** | 18.11 | 2.0x | 100% | Lossless compression; optimal for GPU deployment. |
| **Static INT8** | **9.06** | **4.0x** | 98.88% | Maximum compression; minimal accuracy degradation (<1%). |
| **Dynamic INT8** | 36.16* | 1.0x* | 100% | No storage benefit; compute optimization only. |

*Dynamic INT8 stores weights in FP32 format on disk in PyTorch by default, quantization happens at runtime.*

## 3. Deployment Recommendations

Based on the trade-off analysis:



1.  **Low-Resource Edge (MCU/IoT)**: Deploy **Distilled, Pruned & Quantized Student (Static INT8)**.
    *   **Definition**: The fully optimized student pipeline: Distillation $\to$ Structured Pruning $\to$ Static INT8 Quantization.
    *   **Reference Metrics (ResNet18)**: ~9MB Size (4x reduction), <1% Accuracy Loss.
    *   **Rationale**: Maximizes storage efficiency and integer arithmetic speed suitable for microcontrollers.

2.  **Performance Edge (Jetson/Mobile)**: Deploy **Distilled, Pruned & Quantized Student (FP16)**.
    *   **Reference Metrics (ResNet18)**: ~18MB Size (2x reduction), 0% Accuracy Loss.
    *   **Rationale**: Leverages hardware acceleration for half-precision float (available on modern mobile GPUs) without compromising model fidelity.

3.  **Cloud / Server Scaling**: Deploy **Distilled, Pruned & Quantized Student (FP16)**.
    *   **Reference Metrics (ResNet18)**: ~18MB Size, 0% Accuracy Loss.
    *   **Rationale**: Modern data center GPUs are optimized for FP16 tensor operations. This configuration maximizes throughput (requests/second) while ensuring no degradation in prediction quality.
