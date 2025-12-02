**Repository Layout**
- **`README.md`**: top-level project README (summary). 
- **`DIRECTIONS.md`**: (this file) how to run the pipeline and where outputs live.
- **`src/`**: implementation scripts and modules.
  - `src/data_extraction.py` — builds model-ready artifacts from raw inputs.
  - `src/evaluation/evaluation.py` — metric and plotting helpers (centered + raw metrics).
  - `src/baselines/no_change.py` — baseline B0: predicts 0 for every gene.
  - `src/baselines/global_mean_shift.py` — baseline B1: per-perturbation intercept (panel/global fallback).
  - `src/baselines/additive_control.py` — baseline: average responses from similar TFs (tf_family → batch → global).
- **`data/`**: raw input files. Expected files used by the scripts:
  - `data/Hs27_fibroblast_CRISPRa_mean_pop.h5ad` — AnnData with `obs` (cells) and `var` (genes). Must include a perturbation column (e.g. `perturbation`, `perturbation_id`, `target`, `target_gene`, or `guide`) and, if available, `is_control`, `cell_type`, `batch`, `panel`, `tf_family`.
  - `data/Hs27_gene_sequences_12kb_4912genes.csv` — per-gene sequences with a `gene_id` column used to align genes.
- **`artifacts/`**: generated model inputs (created by `src/data_extraction.py`). Key files:
  - `gene_index.txt` — ordered list of Ensembl gene IDs used as model columns.
  - `targets.parquet` — P × G parquet DataFrame of pseudobulk deltas (ground truth): rows = perturbation IDs, cols = genes from `gene_index`.
  - `covariates.parquet` — per-gene covariates: `seq_length`, `gc`, `baseline_expression`.
- **`results/`**: baseline predictions, per-protocol metrics, and plots (organized per baseline and protocol). Example structure:
  - `results/no_change/loto/` — files for the LOTO protocol from the no-change baseline
  - `results/global_mean_shift/loto/` — files for global mean baseline
  - `results/additive_control/loto/` — files for additive-control baseline


**Inputs — where to get them and where to put them**
- Place your primary AnnData file(s) and the gene sequences CSV under the `data/` directory. The scripts use these default names (you can edit the scripts or call functions if you want different paths):

```bash
# required inputs (default expected paths)
ls data/
# data/Hs27_fibroblast_CRISPRa_mean_pop.h5ad
# data/Hs27_gene_sequences_12kb_4912genes.csv
```

- If you don't have these files yet:
  - The single-cell / perturbation dataset should be produced (or downloaded) as an AnnData `.h5ad` where `adata.obs` contains a perturbation identifier column and ideally `is_control` / `cell_type` / `batch` / `panel` / `tf_family` where available.
  - The gene sequence CSV should include a `gene_id` column (Ensembl IDs) matching `adata.var['gene_id']`.


**what the model predicts**
  - `artifacts/targets.parquet` — for each perturbation p the target vector is
    mean_expr(p) − control_mean (a numeric pseudobulk delta vector of length G).
  - Rows correspond to perturbation identifiers and columns correspond to gene IDs in `artifacts/gene_index.txt`.

- A model should output, for each perturbation p, a vector y_hat_p of length G (one predicted real value per gene). Baselines in `src/baselines/` demonstrate expected output format.


**Model inputs (exact contents & shapes)**

- Artifacts produced by `src/data_extraction.py` (these are the canonical model inputs):
  - `gene_index.txt` — text file with G lines, one Ensembl gene id per line. This determines the column order for all matrices.
  - `artifacts/targets.parquet` — parquet DataFrame P × G (rows = perturbation ids, columns = gene ids in `gene_index.txt`). Values are floats (pseudobulk deltas) and are the ground truth values a model should predict.
  - `artifacts/covariates.parquet` — pandas DataFrame indexed by gene id with columns like `seq_length` (int), `gc` (float), and `baseline_expression` (float). Shape: G × C (C = number of covariates).

- Optional model inputs you may prepare from `data/`:
  - Per-gene sequence features: you may compute embeddings of the sequences in `data/Hs27_gene_sequences_12kb_4912genes.csv` and produce a matrix of shape G × D (D = embedding dim). Ensure rows are ordered by `gene_index.txt`.
  - Per-perturbation metadata (from `adata.obs`) such as `cell_type`, `batch`, `tf_family` — useful for splitting/evaluation and for model covariates.

- Expected formats for model consumers:
  - Targets (ground truth): pandas DataFrame loaded from `artifacts/targets.parquet` with index dtype=str (perturbation ids) and columns exactly equal to the `gene_index` list.
  - Covariates: pandas DataFrame indexed by gene id matching `gene_index`.
  - Model input X (example): to predict P × G you may build X with shape (P, G, F) or flatten to (P, G*F) depending on model. The repository baselines expect model code to output an array of shape (P, G).

Examples:

```python
import pandas as pd
# load canonical artifacts
gene_index = [l.strip() for l in open('artifacts/gene_index.txt')]  # length G
Y = pd.read_parquet('artifacts/targets.parquet')  # shape (P, G)
cov = pd.read_parquet('artifacts/covariates.parquet')  # shape (G, C)

# Ensure ordering
Y = Y.reindex(columns=gene_index)
cov = cov.reindex(index=gene_index)

# If you compute sequence embeddings, make sure they are aligned to gene_index
# embeddings: numpy array of shape (G, D)
```

**How your model should output predictions**

- A model should produce a prediction matrix `preds` shaped (P, G) (same index order as `artifacts/targets.parquet` and same columns as `gene_index.txt`). Write predictions to parquet like the baselines:

```python
import pandas as pd
preds_df = pd.DataFrame(preds, index=Y.index, columns=gene_index)
preds_df.to_parquet('results/my_model/predictions_my_model.parquet')
```



**how to do evaluation**

- Metrics produced (saved under `results/<baseline>/<protocol>/`):
  - Centered metrics: `centered Pearson` (per-pert and mean), `centered RMSE / MAE / MSE` (these demean both true and predicted vectors per perturbation before scoring; use these when you care about within-perturbation pattern recovery).
  - Raw (non-demeaned) metrics: `raw MSE / RMSE / MAE / Pearson` (these capture absolute intercept and magnitude errors).

- Files produced by evaluation routines for a baseline + protocol:
  - `predictions_<baseline>.parquet` — the raw predictions (P × G) written by the baseline.
  - `<baseline>_<metric>_per_pert.csv` — per-perturbation arrays for each metric (e.g., `no_change_raw_mse_per_pert.csv`).
  - `<baseline>_summary.json` — mean metrics and summary values.
  - `raw_mse_hist_<protocol>.png` — zoomed raw MSE histogram (1st–99th percentile view).
  - `pearson_hist_<protocol>.png` — centered Pearson histogram.


**How to call evaluation from your own model**

When you implement a new model, use the evaluation helpers in `src/evaluation/evaluation.py` so outputs are comparable to the baselines. The minimal pattern is:

1. Produce a predictions parquet file `results/<your_model>/predictions_<your_model>.parquet` with the same index and column order as `artifacts/targets.parquet` (index = perturbation ids, columns = `gene_index` in order).

2. Use the evaluation module to compute centered/raw metrics and save outputs. Example:

```python
from pathlib import Path
import pandas as pd
from src.evaluation import evaluation as evalmod

# Paths
targets_path = Path('artifacts/targets.parquet')
preds_path = Path('results/my_model/predictions_my_model.parquet')
out_dir = Path('results/my_model/loto')

# Load and align
Y = pd.read_parquet(targets_path)
preds = pd.read_parquet(preds_path)
# ensure same column order
gene_index = [g.strip() for g in open('artifacts/gene_index.txt')]
Y = Y.reindex(columns=gene_index)
preds = preds.reindex(columns=gene_index)

# Convert to numpy arrays (P, G)
Y_arr = Y.values
preds_arr = preds.values

# Centered metrics (shape recovery)
pear_per, pear_mean = evalmod.centered_pearson_per_pert(Y_arr, preds_arr)
rmse_per, rmse_mean = evalmod.centered_rmse_per_pert(Y_arr, preds_arr)
mse_per, mse_mean = evalmod.centered_mse_per_pert(Y_arr, preds_arr)

# Raw metrics (absolute loss)
raw_mse_per, raw_mse_mean = evalmod.mse_per_pert(Y_arr, preds_arr)

# Save metrics & plots (uses evalmod.save_metrics and plotting helpers)
metrics = {
    'pearson_per_pert': pear_per,
    'pearson_mean': pear_mean,
    'rmse_per_pert': rmse_per,
    'rmse_mean': rmse_mean,
    'mse_per_pert': mse_per,
    'mse_mean': mse_mean,
    'raw_mse_per_pert': raw_mse_per,
    'raw_mse_mean': raw_mse_mean,
}
evalmod.save_metrics(metrics, out_dir=str(out_dir), prefix='my_model_loto')
evalmod.plot_metric_hist(metrics['pearson_per_pert'], str(out_dir / 'pearson_hist_loto.png'), title='MyModel: centered Pearson (LOTO)')
evalmod.plot_metric_hist(metrics['raw_mse_per_pert'], str(out_dir / 'raw_mse_hist_loto.png'), title='MyModel: raw MSE (LOTO)')
```

Notes:
- Always reindex both `Y` and `preds` to the canonical `gene_index` before converting to numpy arrays. Misalignment is the most common source of errors.
- For protocol-specific evaluation (cross-context, family-holdout), you can reuse the baseline scripts' splitting logic or adapt their `load_perturbation_meta()` and `evaluate_and_save()` helpers as examples.
- Use `evalmod.paired_bootstrap_ci()` to compute bootstrap CIs for a scalar metric (we use it in the baselines for centered Pearson mean).



**how to run**

- 1) Build artifacts (targets, covariates, gene index)

```bash
python src/data_extraction.py
# This writes: artifacts/targets.parquet, artifacts/covariates.parquet, artifacts/gene_index.txt
```

- 2) Run baselines (examples)

```bash
# No-change baseline (predicts 0)
python src/baselines/no_change.py

# Global mean-shift baseline (per-pert intercept, Panel fallback -> global)
python src/baselines/global_mean_shift.py

# Additive-control baseline (average responses of similar TFs)
python src/baselines/additive_control.py
```

- 3) Inspect results
  - Per-baseline results are under `results/<baseline>/<protocol>/` (e.g. `results/no_change/loto/`).
  - Open `*_summary.json` for mean metric summaries, and `*_per_pert.csv` files for per-perturbation numbers.
  - Plots: raw MSE histograms are `raw_mse_hist_*.png`; centered Pearson histograms are `pearson_hist_*.png`.