# Gene Expression Prediction Benchmarks

This repository contains a collection of scripts for predicting gene expression under Transcription Factor (TF) perturbations. The project benchmarks various architectures—ranging from Linear Models and XGBoost to Multi-Layer Perceptrons (MLPs) and PEFT Fine-tuning—using embeddings from foundation models like **Nucleotide Transformer (NT-v2)** and **HyenaDNA**.

## Repository Structure

The codebase is organized into four main categories: End-to-End Fine-tuning, MLP Baselines, Linear/Tree Baselines, and Utilities.

### 1. End-to-End Fine-tuning
Directly trains on DNA sequences using Parameter-Efficient Fine-Tuning (PEFT).

- **`NT_v2_ia3_regression.py`**
  - **Model:** Nucleotide Transformer v2 (50M) with IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).
  - **Task:** Regression on TF sequences to predict gene expression changes.
  - **Key Features:** Uses Hugging Face `Trainer`, PEFT configuration, and 5-fold cross-validation. Best for high-performance sequence-to-expression modeling without pre-computed embeddings.

### 2. MLP Models (Embedding-based)
Neural networks trained on pre-computed embeddings to predict expression.

- **`Hs_27_mlp_NT.py`**
  - **Input:** NT-v2 Embeddings.
  - **Architecture:** 3-layer MLP.
  - **Features:** Standard PyTorch training loop with mixed precision (AMP), Early Stopping, and 5-fold CV.

- **`Hs_27_mlp_Hyena.py`**
  - **Input:** HyenaDNA Embeddings.
  - **Architecture:** 3-layer MLP.
  - **Features:** Parallel implementation to the NT version, specifically tuned for benchmarking HyenaDNA representations.

- **`concat_gene_TF_MLP.py`**
  - **Strategy:** Concatenation.
  - **Method:** Instead of treating the problem as matrix factorization (TF $\times$ Gene), this script concatenates the Gene Embedding and TF Embedding into a single vector input for the MLP to predict specific expression values.

### 3. Linear & Tree-based Baselines
Fast, interpretable models to establish performance floors and analyze embedding quality.

- **`LEL_embed.py`** (Linear Embedding Layer - NT Variant)
  - **Method:** Solves the closed-form Ridge regression $Y \approx G_{embed} W P_{embed}^T$.
  - **Distinction:** Uses **pre-trained NT embeddings** for both Genes ($G_{embed}$) and TFs ($P_{embed}$).

- **`Hs27_LEL.py`** (Linear Embedding Layer - PCA Variant)
  - **Method:** Solves a similar closed-form equation.
  - **Distinction:** Derives Gene embeddings via **PCA** on the training expression matrix ($Y$), rather than using language model embeddings.

- **`Hs27_xgboost.py`**
  - **Model:** XGBoost Regressor.
  - **Method:** Trains a gradient-boosted tree on HyenaDNA embeddings. Serves as a strong non-deep-learning baseline.

### 4. Utilities

- **`inference.py`**
  - **Purpose:** Helper script for model loading and inference.
  - **Support:** Handles loading of Nucleotide Transformer v2 (from Hugging Face) and HyenaDNA (from local or repo) models/tokenizers. Useful for generating the embeddings used in the scripts above.

---

## Data Requirements

Most scripts expect the following artifacts (paths may need adjustment in the script headers or via CLI arguments):

1.  **`targets_377.csv`**: The gene expression matrix (Labels).
2.  **`tf_sequences.csv`**: Mapping of TFs to their DNA sequences.
3.  **Embeddings (`.npy`)**: Pre-computed embeddings for NT-v2 or HyenaDNA (required for MLP and Linear scripts).
4.  **`cv_splits_5fold_42.pkl`**: Pickled cross-validation indices to ensure consistent evaluation across all models.

## Usage Examples

**Running the IA3 Fine-tuning:**
```bash
python NT_v2_ia3_regression.py
