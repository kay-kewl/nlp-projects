# Benchmarking DeepSeek LLM vs. Classical ML (SVM/ElasticNet)

## 1. The Problem
While Large Language Models are powerful, they are expensive and can be inconsistent. In specialized tasks like Russian News Named Entity Recognition, BSNLP dataset, or AG News classification, strictly optimized classical methods might offer a better trade-off between cost and accuracy.

## 2. My Approach
*   **Classical Baseline:** Implemented a Logistic Regression with Elastic Net L1+L2 from scratch using Gradient Descent and tuned `scikit-learn` SVMs using TF-IDF vectorization.
*   **LLM Integration:** Designed a zero-shot prompting pipeline for DeepSeek to perform NER.
*   **Analysis:** Compared models on Bias-Variance tradeoff, interpretability, feature weights, and handling of unseen entities.

## 3. Key Findings
*   **High Variance in Classical Models:** Linear models achieved ~0.91 F1 on seen patterns but struggled with generalization due to small training data, High Variance.
*   **LLM Generalization:** DeepSeek with F1 ~0.78 underperformed metrics-wise due to formatting strictness but correctly identified entities unseen in the training set (better generalization).
*   **Optimization:** In the AG News benchmark, BoW + SVM with RBF Kernel slightly outperformed TF-IDF + Logistic Regression, proving that complex kernels are viable for dense text classification.

## 4. Visualizations
*   Includes Confusion Matrices comparing DeepSeek errors vs. SVM errors.
*   Hyperparameter Heatmaps showing the sensitivity of Polynomial SVMs to regularization strength.
