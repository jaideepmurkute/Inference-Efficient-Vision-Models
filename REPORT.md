# Inference Efficient Vision Models: Final Report

## 1. Executive Summary
This project aimed to optimize a Vision Transformer (ViT) for efficient inference on edge devices. We implemented a multi-stage optimization pipeline consisting of **Knowledge Distillation**, **Structured Pruning**, and **Post-Training Quantization (PTQ)**.

**Key Achievements:**
*   **Model Size Reduction**: Reduced the model size from **22MB** (Baseline Student) to **5.2MB** (Pruned + 8-bit Quantized), a **4.2x reduction**.
*   **Accuracy Preservation**: Maintained competitive accuracy (~44%) despite significant compression.
*   **Pipeline Flexibility**: Established a modular pipeline capable of Distillation, Pruning (with Fine-tuning), and Quantization (INT8/FP16) for any compatible ViT architecture.

---

## 2. Methodology

Our optimization pipeline consists of three sequential stages:

### Phase 1: Knowledge Distillation (KD)
We trained a lightweight **Student** model (`vit_tiny_patch16_224`) to mimic a larger **Teacher** model (`vit_base_patch16_224`).
*   **Goal**: Transfer representational power from a heavy model to a compact architecture.
*   **Outcome**: A high-performing baseline student model.

### Phase 2: Structured Pruning
We applied structural pruning to the student model using `torch-pruning`.
*   **Technique**: Removed 20% of channels (Pruning Ratio 0.2) from Linear and Attention layers.
*   **Constraint**: Enforced `round_to=8` to ensure channel dimensions remain hardware-friendly for vectorization.
*   **Recovery**: Implemented a fine-tuning loop to recover accuracy lost during channel removal.
*   **Result**: Reduced parameter count and memory footprint (17.6 MB) while recovering accuracy to **45.00%**.

### Phase 3: Post-Training Quantization (PTQ)
We applied various quantization techniques to the *Pruned* model to further compress it and optimize latency.
*   **Dynamic Quantization (INT8)**: Quantizes weights to 8-bit integers; keeps activations in FP32 (quantized dynamically). Best trade-off for CPUs.
*   **Dynamic Quantization (FP16)**: Casts model weights to Float16.
*   **Static Quantization (INT8)**: Calibrates activations offline. (Note: Proved challenging for this specific ViT architecture without FX-Graph tuning).

---

## 3. Experimental Results

The following table summarizes the performance of the Pruned Model under different quantization schemes on a **standard CPU backend**:

| Quantization Mode | Model Size | Accuracy | Latency (CPU) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 Baseline** | **17.60 MB** | **45.00%** | **~61 ms** | The uncompressed pruned model. Reference capability. |
| **Dynamic INT8** | **5.18 MB** | **43.89%** | **~64 ms** | **Recommended.** 3.4x smaller than Pruned Baseline with minimal accuracy loss (-1%). |
| **Dynamic FP16** | **8.87 MB** | **45.00%** | **~510 ms*** | 2x smaller with perfect accuracy. *High latency on CPU due to lack of native FP16 execution units.* |
| **Static INT8** | **5.04 MB** | **7.50%** | **~40 ms** | Fastest inference, but significantly degrades accuracy for ViTs without advanced calibration techniques (QAT). |

### 4. Key Insights

1.  **INT8 is King for CPU**: Dynamic INT8 Quantization offered the best balance. It compressed the model significantly (down to 5MB) while maintaining 97% of the original accuracy.
2.  **FP16 vs CPU**: While Float16 cuts model size in half (17.6MB -> 8.8MB), standard x86 CPUs lack native FP16 arithmetic instructions. This forces PyTorch to use slow software emulation or casting overhead, resulting in **8x slower inference** (510ms).
    *   *Recommendation*: Use FP16 only for GPU deployments (where it would likely be <10ms) or strictly for storage reduction on disk.
3.  **Structure Matters**: By using "Structured Pruning" instead of random weight zeroing, we ensured that the pruned (17.6MB) model was actually physically smaller and potentially faster, rather than just sparse.

## 5. Conclusion
For deployment on this specific hardware (CPU), the **Pruned + Dynamic INT8** model is the optimal choice. It delivers a **5MB** lightweight vision model that is sufficiently accurate for the target task, representing a massive efficiency gain over the original ViT-Base teacher.
