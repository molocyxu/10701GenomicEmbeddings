"""
No-change baseline pipeline.

Predicts y_p = 0 for all genes and evaluates using utilities in
`src/evaluation/evaluation.py`. Saves metrics, plots, and raw predictions
to `results/no_change/`.
"""
from pathlib import Path
from typing import List
import sys
import logging
import numpy as np
import pandas as pd

# make sure `src` is importable
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from evaluation import evaluation as evalmod
from utils import parquet_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_artifacts(art_dir: str = 'artifacts'):
    art = Path(art_dir)
    gi = art / 'gene_index.txt'
    targets = art / 'targets.parquet'
    tf_emb = art / 'tf_embeddings.parquet'
    
    if not gi.exists() or not targets.exists():
        raise FileNotFoundError('Expected artifacts: gene_index.txt and targets.parquet in artifacts/')
    
    gene_index = [x.strip() for x in gi.read_text().splitlines() if x.strip()]
    Y = parquet_utils.read_targets(targets)
    # ensure columns order
    Y = Y.reindex(columns=gene_index)
    
    # Load TF embeddings if available
    tf_embeddings = None
    try:
        tf_embeddings = parquet_utils.read_tf_embeddings(tf_emb)
        logging.info(f"Loaded TF embeddings shape: {tf_embeddings.shape}")
    except Exception as e:
        logging.info(f"TF embeddings not found; proceeding without them: {e}")
    
    return gene_index, Y, tf_embeddings


def load_perturbation_meta(h5ad_path: str = 'data/Hs27_fibroblast_CRISPRa_mean_pop.h5ad') -> pd.DataFrame:
    try:
        import scanpy as sc
    except Exception:
        logging.warning('scanpy not available; cannot read h5ad for metadata')
        return pd.DataFrame()

    p = Path(h5ad_path)
    if not p.exists():
        logging.info(f'No h5ad found at {h5ad_path}; skipping perturbation metadata load')
        return pd.DataFrame()

    adata = sc.read_h5ad(str(p))
    obs = adata.obs.copy()

    # Determine perturbation column
    group_col_candidates = ['perturbation', 'perturbation_id', 'target', 'target_gene', 'guide']
    group_col = None
    for c in group_col_candidates:
        if c in obs.columns:
            group_col = c
            break

    if group_col is None:
        logging.warning('No perturbation column found in adata.obs')
        return pd.DataFrame()

    # Build metadata per perturbation
    meta = []
    for pert, sub in obs.groupby(group_col):
        row = {'perturbation_id': str(pert)}
        # optional fields
        row['cell_type'] = sub['cell_type'].iloc[0] if 'cell_type' in sub.columns else None
        row['batch'] = sub['batch'].iloc[0] if 'batch' in sub.columns else (sub['panel'].iloc[0] if 'panel' in sub.columns else None)
        # TF family or TF id
        row['tf'] = sub[group_col].iloc[0]
        row['tf_family'] = sub['tf_family'].iloc[0] if 'tf_family' in sub.columns else None
        meta.append(row)

    meta_df = pd.DataFrame(meta).set_index('perturbation_id')
    return meta_df


def evaluate_and_save(Y_df: pd.DataFrame, preds: np.ndarray, meta_df: pd.DataFrame, out_dir: str, protocol_name: str, test_idx: List[str] = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Subset to test_idx if provided
    if test_idx is not None:
        Y_sub = Y_df.loc[test_idx]
        preds_sub = preds[[Y_df.index.get_loc(i) for i in test_idx], :]
    else:
        Y_sub = Y_df
        preds_sub = preds

    Y_arr = Y_sub.values

    pear_per, pear_mean = evalmod.centered_pearson_per_pert(Y_arr, preds_sub)
    rmse_per, rmse_mean = evalmod.centered_rmse_per_pert(Y_arr, preds_sub)
    mae_per, mae_mean = evalmod.centered_mae_per_pert(Y_arr, preds_sub)
    mse_per, mse_mean = evalmod.centered_mse_per_pert(Y_arr, preds_sub)

    metrics = {
        'pearson_per_pert': pear_per,
        'pearson_mean': pear_mean,
        'rmse_per_pert': rmse_per,
        'rmse_mean': rmse_mean,
        'mae_per_pert': mae_per,
        'mae_mean': mae_mean,
    }

    # Also compute RAW (non-demeaned) metrics so constant predictions (like zeros)
    # can be compared by absolute loss to the targets.
    raw_pear = []
    raw_rmse = []
    raw_mae = []
    for t, p in zip(Y_arr, preds_sub):
        t_std = np.std(t, ddof=0)
        p_std = np.std(p, ddof=0)
        if t_std == 0 or p_std == 0:
            raw_pear.append(0.0)
        else:
            cov = np.cov(t, p, ddof=0)[0, 1]
            raw_pear.append(float(cov / (t_std * p_std)))
        raw_rmse.append(float(np.sqrt(np.mean((t - p) ** 2))))
        raw_mae.append(float(np.mean(np.abs(t - p))))

    raw_pear = np.array(raw_pear)
    raw_rmse = np.array(raw_rmse)
    raw_mae = np.array(raw_mae)

    # Also compute raw MSE and centered MSE using evaluation helpers
    raw_mse_per, raw_mse_mean = evalmod.mse_per_pert(Y_arr, preds_sub)

    metrics.update({
        'mse_per_pert': mse_per,
        'mse_mean': mse_mean,
        'raw_pearson_per_pert': raw_pear,
        'raw_pearson_mean': float(np.nanmean(raw_pear)),
        'raw_rmse_per_pert': raw_rmse,
        'raw_rmse_mean': float(np.nanmean(raw_rmse)),
        'raw_mae_per_pert': raw_mae,
        'raw_mae_mean': float(np.nanmean(raw_mae)),
        'raw_mse_per_pert': raw_mse_per,
        'raw_mse_mean': raw_mse_mean,
    })

    # Save metrics and plots
    evalmod.save_metrics(metrics, out_dir=str(out), prefix=f'no_change_{protocol_name}')
    evalmod.plot_metric_hist(metrics['pearson_per_pert'], str(out / f'pearson_hist_{protocol_name}.png'), title=f'No-change: centered Pearson ({protocol_name})')
    # Plot raw MSE (non-demeaned) to capture absolute loss against targets
    if 'raw_mse_per_pert' in metrics:
        vals = np.asarray(metrics['raw_mse_per_pert'])[~np.isnan(metrics['raw_mse_per_pert'])]
        if len(vals) > 0:
            # choose tight x-limits based on percentiles (1st-99th) to zoom into small differences
            p_low, p_high = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
            # add small padding
            pad = max(1e-8, 0.02 * (p_high - p_low))
            xlim = (max(0.0, p_low - pad), p_high + pad)
        else:
            xlim = None
        evalmod.plot_metric_hist(metrics['raw_mse_per_pert'], str(out / f'raw_mse_hist_{protocol_name}.png'), title=f'No-change: raw MSE ({protocol_name})', xlim=xlim)
    else:
        evalmod.plot_metric_hist(metrics['rmse_per_pert'], str(out / f'rmse_hist_{protocol_name}.png'), title=f'No-change: centered RMSE ({protocol_name})')

    # paired bootstrap CI for mean centered Pearson across test perturbations
    def mean_centered_pear(y_t, y_p):
        per, m = evalmod.centered_pearson_per_pert(y_t, y_p)
        return float(np.nanmean(per))

    obs, (l, u) = evalmod.paired_bootstrap_ci(mean_centered_pear, Y_arr, preds_sub, n_boot=200, random_state=0)
    ci_summary = {'pearson_mean': obs, 'pearson_mean_ci_lower': l, 'pearson_mean_ci_upper': u}
    with open(out / f'ci_summary_{protocol_name}.json', 'w') as f:
        import json
        json.dump(ci_summary, f, indent=2)

    logging.info(f'Protocol {protocol_name} - Pearson mean: {pear_mean:.4f}, 95% CI: [{l:.4f}, {u:.4f}]')
    return metrics


def run_no_change(out_dir: str = 'results/no_change'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info('Loading artifacts')
    gene_index, Y_df, tf_embeddings = load_artifacts()
    logging.info(f'Loaded targets shape: {Y_df.shape}')
    if tf_embeddings is not None:
        logging.info(f'Loaded TF embeddings shape: {tf_embeddings.shape}')

    Y = Y_df.values

    # Predictions: zeros (same shape)
    preds = np.zeros_like(Y)
    # Save raw predictions
    preds_df = pd.DataFrame(preds, index=Y_df.index, columns=gene_index)
    preds_path = out / 'predictions_no_change.parquet'
    preds_df.to_parquet(preds_path)
    logging.info(f'Saved predictions to {preds_path}')

    # Load perturbation metadata if available
    meta_df = load_perturbation_meta()

    # Protocol 1: LOTO (evaluate on all perturbations individually)
    logging.info('Running LOTO-style evaluation (per-perturbation)')
    evaluate_and_save(Y_df, preds, meta_df, out_dir=out / 'loto', protocol_name='loto')

    # Protocol 2: Cross-context transfer (by cell_type)
    if 'cell_type' in meta_df.columns:
        logging.info('Running cross-context transfer evaluation by cell_type')
        for ct in meta_df['cell_type'].dropna().unique():
            test_idx = meta_df[meta_df['cell_type'] == ct].index.tolist()
            evaluate_and_save(Y_df, preds, meta_df, out_dir=out / f'cross_context/{ct}', protocol_name=f'cross_context_{ct}', test_idx=test_idx)
    else:
        logging.info('No cell_type metadata available; skipping cross-context evaluation')

    # Protocol 3: Family holdout
    if 'tf_family' in meta_df.columns and meta_df['tf_family'].notna().any():
        logging.info('Running family-holdout evaluation')
        for fam in meta_df['tf_family'].dropna().unique():
            test_idx = meta_df[meta_df['tf_family'] == fam].index.tolist()
            evaluate_and_save(Y_df, preds, meta_df, out_dir=out / f'family_holdout/{fam}', protocol_name=f'family_{fam}', test_idx=test_idx)
    else:
        logging.info('No tf_family metadata available; skipping family-holdout evaluation')

    # Also save overall metrics across all perturbations
    metrics_all = evaluate_and_save(Y_df, preds, meta_df, out_dir=out / 'all', protocol_name='all')

    logging.info('No-change baseline completed. Results saved to ' + str(out.resolve()))
    print('Results saved to', out.resolve())


if __name__ == '__main__':
    run_no_change()
