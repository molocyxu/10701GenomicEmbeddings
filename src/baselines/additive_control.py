"""
Additive control (panel-level) baseline.

Predicts y_p as the average response vector of similar TFs in the training set,
preferring TFs within the same `tf_family`, then falling back to `batch`/`panel`.
If no similar training perturbations exist, falls back to the global mean across
training perturbations. Excludes the held-out perturbation to avoid leakage.

Saves predictions and per-protocol evaluation under `results/additive_control/`.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_artifacts(art_dir: str = 'artifacts'):
    art = Path(art_dir)
    gi = art / 'gene_index.txt'
    targets = art / 'targets.parquet'
    tf_emb = art / 'tf_embeddings.parquet'
    
    if not gi.exists() or not targets.exists():
        raise FileNotFoundError('Expected artifacts: gene_index.txt and targets.parquet in artifacts/')
    
    gene_index = [x.strip() for x in gi.read_text().splitlines() if x.strip()]
    Y = pd.read_parquet(targets)
    Y = Y.reindex(columns=gene_index)
    
    # Load TF embeddings if available
    tf_embeddings = None
    if tf_emb.exists():
        tf_embeddings = pd.read_parquet(tf_emb)
        logging.info(f"Loaded TF embeddings shape: {tf_embeddings.shape}")
    else:
        logging.info("TF embeddings not found; proceeding without them")
    
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

    meta = []
    for pert, sub in obs.groupby(group_col):
        row = {'perturbation_id': str(pert)}
        row['cell_type'] = sub['cell_type'].iloc[0] if 'cell_type' in sub.columns else None
        row['batch'] = sub['batch'].iloc[0] if 'batch' in sub.columns else (sub['panel'].iloc[0] if 'panel' in sub.columns else None)
        row['tf'] = sub[group_col].iloc[0]
        row['tf_family'] = sub['tf_family'].iloc[0] if 'tf_family' in sub.columns else None
        meta.append(row)

    meta_df = pd.DataFrame(meta).set_index('perturbation_id')
    return meta_df


def get_similar_training_idx(pert: str, meta_df: pd.DataFrame, train_idx: List[str]) -> List[str]:
    """Return list of similar training perturbation ids (excluding held_out).

    Preference order:
      1) same tf_family
      2) same batch/panel
      3) empty -> caller should fall back to global mean
    """
    if meta_df.empty or pert not in meta_df.index:
        return []
    fam = meta_df.loc[pert].get('tf_family', None)
    if pd.notna(fam):
        cand = meta_df[(meta_df['tf_family'] == fam)].index.tolist()
        cand = [c for c in cand if c in train_idx]
        if len(cand) > 0:
            return cand

    batch = meta_df.loc[pert].get('batch', None)
    if pd.notna(batch):
        cand = meta_df[(meta_df['batch'] == batch)].index.tolist()
        cand = [c for c in cand if c in train_idx]
        if len(cand) > 0:
            return cand

    return []


def evaluate_and_save(Y_df: pd.DataFrame, preds: np.ndarray, meta_df: pd.DataFrame, out_dir: str, protocol_name: str, test_idx: List[str] = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    raw_mse_per, raw_mse_mean = evalmod.mse_per_pert(Y_arr, preds_sub)

    metrics = {
        'pearson_per_pert': pear_per,
        'pearson_mean': pear_mean,
        'rmse_per_pert': rmse_per,
        'rmse_mean': rmse_mean,
        'mae_per_pert': mae_per,
        'mae_mean': mae_mean,
        'mse_per_pert': mse_per,
        'mse_mean': mse_mean,
        'raw_mse_per_pert': raw_mse_per,
        'raw_mse_mean': raw_mse_mean,
    }

    evalmod.save_metrics(metrics, out_dir=str(out), prefix=f'addctrl_{protocol_name}')
    evalmod.plot_metric_hist(metrics['pearson_per_pert'], str(out / f'pearson_hist_{protocol_name}.png'), title=f'AdditiveControl: centered Pearson ({protocol_name})')
    # Plot raw MSE (non-demeaned) to show absolute loss
    if 'raw_mse_per_pert' in metrics:
        vals = np.asarray(metrics['raw_mse_per_pert'])[~np.isnan(metrics['raw_mse_per_pert'])]
        if len(vals) > 0:
            p_low, p_high = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
            pad = max(1e-8, 0.02 * (p_high - p_low))
            xlim = (max(0.0, p_low - pad), p_high + pad)
        else:
            xlim = None
        evalmod.plot_metric_hist(metrics['raw_mse_per_pert'], str(out / f'raw_mse_hist_{protocol_name}.png'), title=f'AdditiveControl: raw MSE ({protocol_name})', xlim=xlim)
    else:
        evalmod.plot_metric_hist(metrics['rmse_per_pert'], str(out / f'rmse_hist_{protocol_name}.png'), title=f'AdditiveControl: centered RMSE ({protocol_name})')

    # CI
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


def run_additive_control(out_dir: str = 'results/additive_control'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info('Loading artifacts')
    gene_index, Y_df, tf_embeddings = load_artifacts()
    logging.info(f'Loaded targets shape: {Y_df.shape}')
    if tf_embeddings is not None:
        logging.info(f'Loaded TF embeddings shape: {tf_embeddings.shape}')

    meta_df = load_perturbation_meta()

    P, G = Y_df.shape
    preds = np.zeros((P, G), dtype=float)

    logging.info('Running LOTO additive-control baseline')
    for i, pert in enumerate(Y_df.index.tolist()):
        # training set excludes held-out
        train_idx = [x for x in Y_df.index.tolist() if x != pert]
        similar = get_similar_training_idx(pert, meta_df, train_idx)
        if len(similar) > 0:
            vec = Y_df.loc[similar].mean(axis=0).values
        else:
            # fallback to global mean across training
            vec = Y_df.loc[train_idx].mean(axis=0).values
        preds[i, :] = vec

    preds_df = pd.DataFrame(preds, index=Y_df.index, columns=gene_index)
    preds_path = out / 'predictions_addctrl.parquet'
    preds_df.to_parquet(preds_path)
    logging.info(f'Saved predictions to {preds_path}')

    # Protocol evaluations
    evaluate_and_save(Y_df, preds, meta_df, out_dir=out / 'loto', protocol_name='loto')
    # cross-context by cell_type
    if 'cell_type' in meta_df.columns:
        for ct in meta_df['cell_type'].dropna().unique():
            test_idx = meta_df[meta_df['cell_type'] == ct].index.tolist()
            preds_ct = preds.copy()
            for pert in test_idx:
                train_idx = [x for x in Y_df.index.tolist() if x not in test_idx]
                similar = get_similar_training_idx(pert, meta_df.drop(index=test_idx) if not meta_df.empty else meta_df, train_idx)
                if len(similar) > 0:
                    vec = Y_df.loc[similar].mean(axis=0).values
                else:
                    vec = Y_df.loc[train_idx].mean(axis=0).values
                preds_ct[Y_df.index.get_loc(pert), :] = vec
            evaluate_and_save(Y_df, preds_ct, meta_df, out_dir=out / f'cross_context/{ct}', protocol_name=f'cross_context_{ct}', test_idx=test_idx)
    else:
        logging.info('No cell_type metadata; skipping cross-context evaluation')

    # family holdout
    if 'tf_family' in meta_df.columns and meta_df['tf_family'].notna().any():
        for fam in meta_df['tf_family'].dropna().unique():
            test_idx = meta_df[meta_df['tf_family'] == fam].index.tolist()
            preds_fam = preds.copy()
            for pert in test_idx:
                train_idx = [x for x in Y_df.index.tolist() if x not in test_idx]
                similar = get_similar_training_idx(pert, meta_df.drop(index=test_idx) if not meta_df.empty else meta_df, train_idx)
                if len(similar) > 0:
                    vec = Y_df.loc[similar].mean(axis=0).values
                else:
                    vec = Y_df.loc[train_idx].mean(axis=0).values
                preds_fam[Y_df.index.get_loc(pert), :] = vec
            evaluate_and_save(Y_df, preds_fam, meta_df, out_dir=out / f'family_holdout/{fam}', protocol_name=f'family_{fam}', test_idx=test_idx)
    else:
        logging.info('No tf_family metadata; skipping family-holdout evaluation')

    evaluate_and_save(Y_df, preds, meta_df, out_dir=out / 'all', protocol_name='all')

    logging.info('Additive-control baseline completed. Results saved to ' + str(out.resolve()))
    print('Results saved to', out.resolve())


if __name__ == '__main__':
    run_additive_control()
