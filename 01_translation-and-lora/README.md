# Machine Translation & Efficient Fine-Tuning with LoRA

## 1. The Problem
Machine translation requires handling long-range dependencies in sequences. Furthermore, fine-tuning large pre-trained models, like BERT or RoBERTa, for downstream tasks, like NER, is computationally expensive.

## 2. My Approach
This project is divided into two architectural implementations:
1.  **Seq2Seq with Attention:** a GRU-based Encoder-Decoder from scratch using `torch.nn.Module` with no high-level abstractions. I implemented an Additive Attention mechanism to allow the decoder to focus on specific parts of the source sentence.
2.  **Custom LoRA Implementation:** To optimize fine-tuning, I implemented Low-Rank Adaptation, LoRA, from scratch. Instead of updating all weights $W$, I injected rank-decomposition matrices $A$ and $B$ such that $\Delta W = BA$, reducing trainable parameters by >99%.

## 3. Key Results
*   **Attention Maps:** Visualized alignment between Russian and English tokens;
*   **LoRA Efficiency:** Achieved comparable F1 scores on the Multinerd dataset while training only 0.18% of the parameters compared to full fine-tuning.

## 4. File Structure
*   `src/attention_model.py`: Custom Encoder-Decoder and Attention classes;
*   `src/custom_lora.py`: Manual implementation of the LoRA adapter layer.
