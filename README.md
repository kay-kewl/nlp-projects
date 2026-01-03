# NLP & LLM Optimization

A collection of Natural Language Processing projects focusing on the transition from classical statistical methods to modern Large Language Models. This repository demonstrates implementations of core architectures from scratch, Seq2Seq, Attention, LoRA, and benchmarks them against industry standards.

## Projects

### 1. [Machine Translation & Efficient Fine-Tuning](./01_translation-and-lora)
*   **Highlights:**
    *   Implemented a GRU-based Seq2Seq model with Additive Attention from scratch;
    *   Built Low-Rank Adaptation, LoRA, layers manually to fine-tune BERT for NER, reducing trainable parameters by >99%;
    *   Visualized attention alignment maps using Bokeh.

### 2. [LLM vs. Classical ML Benchmarks](./02_llm-vs-classical-benchmarks)
*   **Highlights:**
    *   Benchmarked zero-shot Deepseek LLM performance against supervised linear models on Russian Named Entity Recognition;
    *   Implemented Logistic Regression with Elastic Net regularization using gradient descent;
    *   Conducted extensive bias-variance analysis and error analysis on the AG News dataset.

## Setup

1. Clone the repository:
  ```bash
    git clone https://github.com/kay-kewl/nlp-projects.git
    cd nlp-projects
  ```

2. Create and activate a virtual environment:
  ```bash
    python -m venv venv

    # windows 
    .\venv\Scripts\activate
    # macos/linux
    source venv/bin/activate
  ```
  
3. Install dependencies:
  ```bash
    pip install -r requirements.txt
  ```
