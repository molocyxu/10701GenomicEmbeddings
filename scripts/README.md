# Model Training & Evaluation Scripts

This directory contains the core training and evaluation scripts for gene expression prediction tasks. These scripts cover a range of methodologies, from simple linear baselines and XGBoost to Transformer-based (Nucleotide Transformer, HyenaDNA) MLP heads and end-to-end PEFT fine-tuning.

## Directory Structure

### 1. End-to-End Fine-tuning
Direct training on DNA sequences using Parameter-Efficient Fine-Tuning techniques.

* **`NT_v2_ia3_regression.py`**
    * **Model:** Nucleotide Transformer v2 (50M) + IA3 (PEFT).
    * **Function:** Takes TF sequences and directly performs regression to predict gene expression changes.
    * **Features:** Utilizes the Hugging Face `Trainer` framework and supports 5-fold cross-validation.

### 2. MLP Models
Lightweight neural networks (3-layer MLPs) trained on pre-computed embeddings.

* **`Hs_27_mlp_NT.py`**
    * **Input:** Embeddings extracted from NT-v2.
    * **Architecture:** Standard 3-layer MLP, supporting mixed precision training (AMP) and early stopping.
* **`Hs_27_mlp_Hyena.py`**
    * **Input:** Embeddings extracted from HyenaDNA.
    * **Architecture:** Identical structure to the NT version, designed for fair comparison of HyenaDNA representations.
* **`concat_gene_TF_MLP.py`**
    * **Strategy:** Concatenation.
    * **Method:** Concatenates Gene Embeddings and TF Embeddings to feed into an MLP, rather than treating the problem as a matrix factorization task.

### 3. Linear & Tree-based Baselines
Lightweight models used to establish performance baselines and validate embedding quality.

* **`LEL_embed.py`** (Linear Embedding Layer - NT Variant)
    * **Method:** Closed-form Ridge regression ($Y \approx G W P^T$).
    * **Features:** Uses **NT-v2 Embeddings** for both Genes and TFs.
* **`Hs27_LEL.py`** (Linear Embedding Layer - PCA Variant)
    * **Method:** Closed-form regression as above.
    * **Features:** Gene Embeddings are obtained via **PCA** on the expression matrix $Y$, while TFs use embeddings.
* **`Hs27_xgboost.py`**
    * **Model:** XGBoost Regressor.
    * **Features:** Gradient boosted tree regression model based on HyenaDNA Embeddings, providing a strong non-deep learning baseline.

### 4. Utilities

* **`inference.py`**
    * **Function:** Generic interface for model loading and inference.
    * **Support:** Loads Hugging Face NT-v2 models and local/remote HyenaDNA models; commonly used to pre-calculate embeddings for the scripts above.

---

## Usage

Most training scripts require the following data files (paths should be configured in the script header or via arguments):
1. `targets_377.csv`: Gene expression matrix (Labels).
2. `tf_sequences.csv`: TF sequence information.
3. `cv_splits_5fold_42.pkl`: Pre-defined 5-fold cross-validation indices.
4. `*.npy`: Pre-computed embeddings (required for MLP and linear models only).

### Running Examples

**Run IA3 Fine-tuning:**
```bash
python NT_v2_ia3_regression.py
