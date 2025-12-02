"""
Data extraction utilities

Produces the following artifacts in `artifacts/`:
- `gene_index.txt` : canonical ordered list of gene_ids
- `targets.parquet` : per-perturbation pseudo-bulk deltas (P x G)
- `covariates.parquet` : per-gene covariates (baseline expression, gc, seq_length)

Usage: run as a script or call main() from notebooks.
"""
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm


# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_sequences(seq_csv: str) -> pd.DataFrame:
    df = pd.read_csv(seq_csv)
    return df


def load_adata(h5ad_path: str):
    adata = sc.read_h5ad(h5ad_path)
    return adata


def gc_fraction(seq: str) -> float:
    s = str(seq).upper()
    if len(s) == 0:
        return 0.0
    return (s.count('G') + s.count('C')) / len(s)


def build_gene_index(adata, seq_df: pd.DataFrame) -> list:
    if 'gene_id' in adata.var.columns:
        var_genes = adata.var['gene_id'].astype(str).tolist()
    else:
        var_genes = adata.var_names.astype(str).tolist()

    seq_genes = seq_df['gene_id'].astype(str).tolist()

    # Keep order of adata.var but restrict to genes with sequences present
    gene_index = [g for g in var_genes if g in seq_genes]
    return gene_index


def compute_pseudobulk_deltas(adata, gene_index, group_col_candidates=None, normalize=True):
    if group_col_candidates is None:
        group_col_candidates = ['perturbation', 'perturbation_id', 'target', 'target_gene', 'guide']

    obs = adata.obs
    group_col = None
    for c in group_col_candidates:
        if c in obs.columns:
            group_col = c
            break

    if group_col is None:
        raise ValueError('No perturbation column found in adata.obs. Expected one of: ' + ','.join(group_col_candidates))

    logger = logging.getLogger(__name__)
    logger.info(f"Building pseudobulk deltas using group column: {group_col}")

    # Convert to dense DataFrame (obs x var). Use to_df() for convenience.
    X = adata.to_df()
    # If adata.var contains Ensembl gene IDs in 'gene_id', map columns to those IDs
    if 'gene_id' in adata.var.columns:
        try:
            gene_id_map = adata.var['gene_id']
            X_cols_mapped = [str(gene_id_map.loc[c]) for c in X.columns]
            X.columns = X_cols_mapped
        except Exception:
            # fallback: just use var_names order mapped by position
            X.columns = [str(g) for g in adata.var['gene_id'].astype(str).tolist()]

    # find control cells
    control_mask = None
    if 'is_control' in obs.columns:
        control_mask = obs['is_control'].astype(bool).values
    else:
        low = obs[group_col].astype(str).str.lower()
        control_mask = low.str.contains('control') | low.str.contains('ntc') | low.str.contains('non-target')

    num_controls = int(control_mask.sum()) if hasattr(control_mask, 'sum') else 0
    logger.info(f"Found {num_controls} control cells (fallback to global mean if 0)")

    if num_controls == 0:
        control_mean = X.mean(axis=0)
    else:
        control_mean = X.loc[control_mask, :].mean(axis=0)

    pert_vals = []
    pert_meta = []
    groups = obs[group_col].astype(str).unique().tolist()

    for g in tqdm(groups, desc="Processing perturbations"):
        if str(g).lower() in ['control', 'ntc', 'nan', 'nan.0', 'none']:
            continue
        mask = obs[group_col].astype(str) == g
        if mask.sum() == 0:
            continue
        mean_expr = X.loc[mask, :].mean(axis=0)
        delta = mean_expr - control_mean
        # align to gene_index (Series.reindex expects index labels)
        delta = delta.reindex(gene_index)
        pert_vals.append(delta.values)
        pert_meta.append({'perturbation_id': str(g)})

    if len(pert_vals) == 0:
        raise ValueError('No perturbations found to build targets.')

    Y = pd.DataFrame(np.vstack(pert_vals), columns=gene_index)
    meta = pd.DataFrame(pert_meta)
    Y.index = meta['perturbation_id'].values

    return Y, control_mean.reindex(index=gene_index)


def compute_covariates(seq_df: pd.DataFrame, baseline_expression: pd.Series) -> pd.DataFrame:
    seq_df = seq_df.set_index('gene_id')
    df = pd.DataFrame(index=seq_df.index)
    df['seq_length'] = seq_df['sequence'].str.len()
    df['gc'] = seq_df['sequence'].apply(gc_fraction)
    df['baseline_expression'] = baseline_expression.reindex(df.index).values
    return df


def save_artifacts(gene_index, Y: pd.DataFrame, covariates: pd.DataFrame, out_dir: str = 'artifacts'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gi_path = out / 'gene_index.txt'
    with gi_path.open('w') as f:
        for g in gene_index:
            f.write(f"{g}\n")

    targets_path = out / 'targets.parquet'
    Y.to_parquet(targets_path)

    cov_path = out / 'covariates.parquet'
    covariates.to_parquet(cov_path)

    print(f"Saved gene_index ({len(gene_index)}), targets -> {targets_path}, covariates -> {cov_path}")


def main(h5ad_path='data/Hs27_fibroblast_CRISPRa_mean_pop.h5ad', seq_csv='data/Hs27_gene_sequences_12kb_4912genes.csv', out_dir='artifacts'):
    logging.info(f"Loading AnnData from {h5ad_path}")
    adata = load_adata(h5ad_path)
    logging.info(f"Loading sequences from {seq_csv}")
    seq_df = load_sequences(seq_csv)

    logging.info("Building gene index (intersecting adata.var and sequences)")
    gene_index = build_gene_index(adata, seq_df)

    logging.info(f"Computing pseudobulk targets for {len(gene_index)} genes")
    Y, baseline = compute_pseudobulk_deltas(adata, gene_index)

    logging.info("Computing covariates (GC, seq length, baseline expression)")
    covariates = compute_covariates(seq_df[seq_df['gene_id'].isin(gene_index)].copy(), baseline)

    logging.info(f"Saving artifacts to {out_dir}")
    save_artifacts(gene_index, Y, covariates, out_dir=out_dir)


if __name__ == '__main__':
    main()
