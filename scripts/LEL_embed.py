#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-fold linear model using NT gene/TF embeddings instead of PCA.

- Input:
    - Gene expression matrix (perturbations × genes), e.g. artifacts/targets_*.csv/parquet
    - NT-v2 gene & TF embeddings: NTv2_50m_gene_embeddings.npy, NTv2_50m_gene_ids.npy,
                                  NTv2_50m_tf_embeddings.npy,   NTv2_50m_tf_ids.npy
    - tf_sequences.csv: for mapping TF IDs <-> gene names (to align perturbations with TF embeddings)

- Model:
    Let:
        Y_train: G × N_train  (rows = genes, cols = perturbations)
        G_embed: G × K        (NT gene embeddings, aligned to gene_index)
        P_train: N_train × K  (NT TF embeddings for each perturbation in train set)

    We approximate:
        Y_center ≈ G_embed W P_train^T
    with ridge on both G_embed and P_train, using the same closed form structure as in the paper
    (now K = embedding dim, e.g. 512).

    Closed-form:
        GTG = G^T G
        GTY = G^T Y_center
        PTP = P^T P
        A   = (GTG + λI)^(-1) GTY
        W   = A P (PTP + λI)^(-1)

    Prediction for test set:
        Ŷ_test = G_embed W P_test^T + b

"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt

def plot_pred_vs_gt(y_true, y_pred, out_path, title="Pred vs GT"):
    # Flatten arrays
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    # Remove NaNs
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    
    # Keep only data points within [-1, 1] range for both axes
    range_mask = (yt >= -1.0) & (yt <= 1.0) & (yp >= -1.0) & (yp <= 1.0)
    yt = yt[range_mask]
    yp = yp[range_mask]
    
    if len(yt) == 0:
        return

    # Calculate global correlation for the plot title
    from scipy.stats import pearsonr
    corr, _ = pearsonr(yt, yp)
    
    plt.figure(figsize=(8, 6))
    # Use hexbin for large datasets
    plt.hexbin(yt, yp, gridsize=50, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(label='log10(count)')
    
    # Add diagonal line
    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.xlabel('Ground Truth Expression')
    plt.ylabel('Predicted Expression')
    plt.title(f"{title}\nGlobal Pearson r = {corr:.4f}")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# =========================
# Argument parsing
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold linear model with NT embeddings")

    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root directory (default: current directory). "
             "Should contain artifacts/ and data/tf_sequences.csv.",
    )
    parser.add_argument(
        "--targets-path",
        type=str,
        default=None,
        help="Optional override for artifacts/targets_*.csv or .parquet",
    )
    parser.add_argument(
        "--gene-index",
        type=str,
        default=None,
        help="Optional override for artifacts/gene_index.txt",
    )
    parser.add_argument(
        "--tf-seq-csv",
        type=str,
        default=None,
        help="Optional override for data/tf_sequences.csv for mapping perturbations -> TF gene_id/gene_name",
    )
    parser.add_argument(
        "--ntv2-dir",
        type=str,
        default="NT-v2-50M",
        help="Directory containing NT-v2 embeddings npy files "
             "(NTv2_50m_gene_embeddings.npy, NTv2_50m_gene_ids.npy, "
             " NTv2_50m_tf_embeddings.npy, NTv2_50m_tf_ids.npy).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="linear_nt_embed",
        help="Subdirectory under results/ for outputs (default: linear_nt_embed)",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Optional cap on number of read-out genes (columns) - takes first N.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=None,
        help="Select top N genes with highest mean absolute expression.",
    )
    parser.add_argument(
        "--max-perturbations",
        type=int,
        default=None,
        help="Optional cap on number of perturbations (rows).",
    )
    parser.add_argument(
        "--lambda-ridge",
        type=float,
        default=0.1,
        help="Ridge regularization λ (default: 0.1).",
    )
    parser.add_argument(
        "--cv-splits-file",
        type=str,
        default=None,
        help="Path to save/load 5-fold splits (for reproducibility across models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Bootstrap samples for CI (default: 200).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress during 5-fold training.",
    )

    return parser.parse_args()


# =========================
# Path / data loading helpers
# =========================

def resolve_repo_root(base_hint: str) -> Path:
    here = Path(__file__).resolve().parent
    candidate = Path(base_hint).expanduser()
    if not candidate.is_absolute():
        candidate = (here / candidate).resolve()

    if (candidate / "artifacts").exists():
        return candidate
    elif (candidate / "data" / "artifacts").exists():
        return candidate / "data"
    else:
        raise FileNotFoundError(
            f"Could not find artifacts/ under {candidate} or {candidate}/data"
        )


def load_targets_and_genes(repo_root: Path, args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load targets (perturbation × genes) and gene_index (list of read-out genes).
    """
    art_dir = repo_root / "artifacts"

    # targets
    if args.targets_path:
        targets_path = Path(args.targets_path).expanduser().resolve()
    else:
        # default name
        default_targets = art_dir / "targets_377_overlap.csv"
        targets_path = default_targets

    suffix = targets_path.suffix.lower()
    if suffix == ".csv":
        targets = pd.read_csv(targets_path, index_col=0)
    elif suffix in {".parquet", ".pq"}:
        targets = pd.read_parquet(targets_path)
    else:
        raise ValueError(f"Unsupported targets file format: {targets_path}")

    # gene index
    if args.gene_index:
        gene_index_path = Path(args.gene_index).expanduser().resolve()
    else:
        gene_index_path = art_dir / "gene_index.txt"

    gene_index = [g.strip() for g in gene_index_path.read_text().splitlines() if g.strip()]

    # reindex columns to gene_index order
    targets = targets.reindex(columns=gene_index)

    # remove duplicate perturbation IDs just in case
    if targets.index.duplicated().any():
        targets = targets[~targets.index.duplicated(keep="first")]

    # optional caps
    if args.max_perturbations is not None:
        targets = targets.iloc[: args.max_perturbations, :]

    if args.max_genes is not None:
        gene_index = gene_index[: args.max_genes]
        targets = targets.loc[:, gene_index]

    if args.n_top_genes is not None:
        # Compute mean absolute expression per gene
        gene_importance = np.abs(targets.values).mean(axis=0)
        if len(gene_importance) > args.n_top_genes:
            # Get indices of top N genes
            top_indices = np.argsort(gene_importance)[-args.n_top_genes:]
            # Keep them in original order
            top_indices = np.sort(top_indices)
            
            # Subset gene_index and targets
            selected_genes = [gene_index[i] for i in top_indices]
            gene_index = selected_genes
            targets = targets.iloc[:, top_indices]
            print(f"[info] Selected top {args.n_top_genes} genes by mean absolute expression.")

    return targets, gene_index


def load_nt_embeddings(
    repo_root: Path,
    args: argparse.Namespace,
    gene_index: List[str],
    perturbations: pd.Index,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Load NT-v2 gene & TF embeddings, align:
        - genes to gene_index
        - perturbations to TF embeddings via tf_sequences.csv

    Returns:
        gene_emb_matrix: (G, K)
        tf_emb_matrix:   (P_aligned, K)
        aligned_perturbations: Index of perturbations kept
    """
    # Determine NT-v2 dir
    ntv2_dir = Path(args.ntv2_dir).expanduser().resolve()
    if not ntv2_dir.exists():
        # Maybe relative to project root
        project_root = repo_root.parent if repo_root.name == "data" else repo_root
        ntv2_dir = (project_root / args.ntv2_dir).expanduser().resolve()
        if not ntv2_dir.exists():
            raise FileNotFoundError(f"NT-v2 directory not found: {args.ntv2_dir}")

    # Load npy files
    gene_embeddings = np.load(ntv2_dir / "NTv2_50m_gene_embeddings.npy")  # (N_gene_all, K)
    gene_ids = np.load(ntv2_dir / "NTv2_50m_gene_ids.npy")                # (N_gene_all,)
    tf_embeddings = np.load(ntv2_dir / "NTv2_50m_tf_embeddings.npy")      # (N_tf_all, K)
    tf_ids = np.load(ntv2_dir / "NTv2_50m_tf_ids.npy")                    # (N_tf_all,)

    print(f"[info] Loaded NT-v2 embeddings: {tf_embeddings.shape[0]} TFs × {tf_embeddings.shape[1]} dims, "
          f"{gene_embeddings.shape[0]} genes × {gene_embeddings.shape[1]} dims")

    # Align genes: gene_ids (from NT) vs gene_index (read-out genes)
    gene_id_to_row = {str(gid): i for i, gid in enumerate(gene_ids)}
    gene_rows = []
    missing_genes = []
    for g in gene_index:
        key = str(g)
        if key in gene_id_to_row:
            gene_rows.append(gene_id_to_row[key])
        else:
            missing_genes.append(g)
    if missing_genes:
        raise ValueError(
            f"{len(missing_genes)} read-out genes not found in NT gene_ids. Examples: {missing_genes[:5]}"
        )
    gene_emb_matrix = gene_embeddings[gene_rows, :]  # (G, K)

    # Map TF ids to gene_name via tf_sequences.csv, then align to perturbations
    if args.tf_seq_csv:
        tf_seq_path = Path(args.tf_seq_csv).expanduser().resolve()
    else:
        project_root = repo_root.parent if repo_root.name == "data" else repo_root
        tf_seq_path = project_root / "data" / "tf_sequences.csv"

    if not tf_seq_path.exists():
        raise FileNotFoundError(f"TF sequence CSV not found at {tf_seq_path}. Provide --tf-seq-csv.")

    tf_df = pd.read_csv(tf_seq_path)
    id_to_name = {}
    for _, row in tf_df.iterrows():
        gid = str(row["gene_id"]).strip()
        gname = str(row["gene_name"]).strip()
        if gid and gname:
            id_to_name[gid] = gname

    # Build TF embedding DataFrame indexed by TF names (or ids if name missing)
    tf_names = [id_to_name.get(str(tid), str(tid)) for tid in tf_ids]
    tf_emb_df = pd.DataFrame(
        tf_embeddings,
        index=[str(n).upper() for n in tf_names],
        columns=[f"emb_{i}" for i in range(tf_embeddings.shape[1])],
    )

    # Align to perturbations (which may be gene_name or gene_id)
    pert_index_upper = pd.Index([str(p).upper() for p in perturbations])
    tf_emb_df = tf_emb_df[~tf_emb_df.index.duplicated(keep="first")]

    aligned_mask = pert_index_upper.isin(tf_emb_df.index)
    aligned_perturbations = perturbations[aligned_mask]

    if (~aligned_mask).sum() > 0:
        missing_perts = perturbations[~aligned_mask]
        print(f"[warn] Dropping {len(missing_perts)} perturbations without NT TF embeddings.")
        print(f"       examples: {list(missing_perts[:5])}")

    tf_emb_matrix = tf_emb_df.loc[pert_index_upper[aligned_mask]].values.astype(np.float32)

    return gene_emb_matrix.astype(np.float32), tf_emb_matrix, aligned_perturbations


# =========================
# 5-fold split save / load
# =========================

def save_cv_splits(splits_data: dict, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(splits_data, f)
    print(f"[info] Saved CV splits to {filepath}")


def load_cv_splits(filepath: str) -> dict:
    with open(filepath, "rb") as f:
        splits_data = pickle.load(f)
    print(f"[info] Loaded CV splits from {filepath}")
    return splits_data


# =========================
# Evaluation metrics
# =========================

def centered_pearson_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Per-perturbation Pearson correlation after centering each perturbation profile
    (subtract its own mean across genes).
    """
    from scipy.stats import pearsonr

    P = Y_true.shape[0]
    corrs = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 1:
            yt_c = yt[mask] - yt[mask].mean()
            yp_c = yp[mask] - yp[mask].mean()
            r, _ = pearsonr(yt_c, yp_c)
            corrs[i] = r if np.isfinite(r) else 0.0
        else:
            corrs[i] = 0.0
    return corrs, float(np.nanmean(corrs))


def centered_rmse_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    P = Y_true.shape[0]
    rmses = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 0:
            yt_c = yt[mask] - yt[mask].mean()
            yp_c = yp[mask] - yp[mask].mean()
            rmses[i] = np.sqrt(np.mean((yt_c - yp_c) ** 2))
        else:
            rmses[i] = np.nan
    return rmses, float(np.nanmean(rmses))


def centered_mae_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    P = Y_true.shape[0]
    maes = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 0:
            yt_c = yt[mask] - yt[mask].mean()
            yp_c = yp[mask] - yp[mask].mean()
            maes[i] = np.mean(np.abs(yt_c - yp_c))
        else:
            maes[i] = np.nan
    return maes, float(np.nanmean(maes))


def centered_mse_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    P = Y_true.shape[0]
    mses = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 0:
            yt_c = yt[mask] - yt[mask].mean()
            yp_c = yp[mask] - yp[mask].mean()
            mses[i] = np.mean((yt_c - yp_c) ** 2)
        else:
            mses[i] = np.nan
    return mses, float(np.nanmean(mses))


def mse_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    P = Y_true.shape[0]
    mses = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 0:
            mses[i] = np.mean((yt[mask] - yp[mask]) ** 2)
        else:
            mses[i] = np.nan
    return mses, float(np.nanmean(mses))


def l2_loss_per_pert(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    L2 距离（欧氏范数） per perturbation:
      ||y_true - y_pred||_2
    """
    P = Y_true.shape[0]
    vals = np.zeros(P, dtype=float)
    for i in range(P):
        yt = Y_true[i]
        yp = Y_pred[i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 0:
            vals[i] = np.linalg.norm(yt[mask] - yp[mask], ord=2)
        else:
            vals[i] = np.nan
    return vals, float(np.nanmean(vals))


def save_metrics_json(metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()},
            f,
            indent=2,
        )


def paired_bootstrap_ci(func, Y_true: np.ndarray, Y_pred: np.ndarray, n_boot: int = 200, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    P = Y_true.shape[0]
    obs = func(Y_true, Y_pred)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, P, size=P)
        boot_vals.append(func(Y_true[idx], Y_pred[idx]))
    boot_vals = np.array(boot_vals)
    lo = float(np.percentile(boot_vals, 2.5))
    hi = float(np.percentile(boot_vals, 97.5))
    return float(obs), (lo, hi)


# =========================
# Linear model: one fold (NT embeddings)
# =========================

def train_single_fold_linear_nt(
    fold_idx: int,
    Y: np.ndarray,           # (P, G)
    gene_emb: np.ndarray,    # (G, K)
    tf_emb: np.ndarray,      # (P, K)
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    lam: float = 0.1,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在一个 fold 上训练 NT-embedding linear 模型并预测 test perturbations。

    - Y: (P, G) perturbation × gene  (已填好 NaN)
    - gene_emb: (G, K) NT gene embeddings aligned to gene_index
    - tf_emb: (P, K)  NT TF embeddings aligned to perturbations (subset after dropping missing)
    """

    P_total, G = Y.shape
    G_count, K = gene_emb.shape
    assert G == G_count, f"Gene dim mismatch: Y has {G}, gene_emb has {G_count}"

    # Y_train_full: (G, N_train)
    Y_train_full = Y[train_indices, :].T  # genes × perturb

    # 基因均值 b (G × 1)
    b = Y_train_full.mean(axis=1, keepdims=True)

    # 中心化
    Y_center_full = Y_train_full - b  # (G, N_train)

    # P_train: (N_train, K)
    P_train = tf_emb[train_indices, :]  # TF embeddings for training perturbations

    # 构造闭式解中的各项
    G_embed = gene_emb  # (G, K)

    GTG = G_embed.T @ G_embed              # (K, K)
    GTY = G_embed.T @ Y_center_full        # (K, N_train)
    PTP = P_train.T @ P_train              # (K, K)

    lamI_G = lam * np.eye(K, dtype=np.float64)
    lamI_P = lam * np.eye(K, dtype=np.float64)

    # A = (G^T G + λI)^(-1) G^T Y_center
    A = np.linalg.solve(GTG + lamI_G, GTY)         # (K, N_train)
    # W_temp = A P_train
    W_temp = A @ P_train                           # (K, K)
    # W = W_temp (P^T P + λI)^(-1)
    W = W_temp @ np.linalg.inv(PTP + lamI_P)       # (K, K)

    # 对 test 集预测
    P_test = tf_emb[test_indices, :]               # (N_test, K)
    # Ŷ = G_embed W P_test^T + b   → (G, N_test)
    Y_hat = G_embed @ W @ P_test.T + b             # (G, N_test)
    preds_test = Y_hat.T                           # (N_test, G)

    # Training predictions
    P_train = tf_emb[train_indices, :]
    Y_hat_train = G_embed @ W @ P_train.T + b
    preds_train = Y_hat_train.T

    return (
        fold_idx,
        test_indices,
        preds_test.astype(np.float32),
        train_indices,
        preds_train.astype(np.float32),
    )


# =========================
# 5-fold CV wrapper
# =========================

def five_fold_cv_linear_nt(
    Y: np.ndarray,
    gene_emb: np.ndarray,
    tf_emb: np.ndarray,
    args: argparse.Namespace,
    use_progress: bool = False,
    cv_splits_file: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    5-fold CV over perturbations using NT-embedding-based linear model.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    P, G = Y.shape
    preds = np.full((P, G), np.nan, dtype=np.float32)
    train_preds = np.full((P, G), np.nan, dtype=np.float32)
    train_filled = np.zeros(P, dtype=bool)

    # preprocess NaN: 用基因均值填充（safe_Y）
    finite_mask = np.isfinite(Y)
    safe_Y = np.zeros_like(Y, dtype=np.float32)
    for g in range(G):
        gene_finite = finite_mask[:, g]
        if gene_finite.any():
            baseline = float(np.nanmean(Y[:, g]))
            safe_Y[:, g] = np.where(gene_finite, Y[:, g], baseline)
        else:
            safe_Y[:, g] = 0.0

    # 复用或生成 5-fold splits
    fold_splits = None
    if cv_splits_file and Path(cv_splits_file).exists():
        splits_data = load_cv_splits(cv_splits_file)
        stored_total = splits_data.get("total_samples")
        if stored_total == P and splits_data.get("method") == "5fold":
            fold_splits = splits_data["fold_splits"]
            print(f"[info] Using existing 5-fold splits with {len(fold_splits)} folds.")
        else:
            print(
                f"[warn] Stored 5-fold splits expect {stored_total} samples or method mismatch; "
                f"regenerating splits."
            )

    if fold_splits is None:
        kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(np.arange(P)))
        if cv_splits_file:
            splits_data = {
                "method": "5fold",
                "n_splits": 5,
                "random_state": args.seed,
                "fold_splits": fold_splits,
                "total_samples": P,
            }
            save_cv_splits(splits_data, cv_splits_file)

    print(f"[info] Running 5-fold CV with NT-embedding linear model (λ={args.lambda_ridge})...")

    if use_progress and tqdm is not None:
        pbar = tqdm(total=5, desc="Folds", unit="fold")
    else:
        pbar = None

    for fold_idx, (train_indices, test_indices) in enumerate(fold_splits):
        if pbar is not None:
            pbar.set_postfix({"fold": f"{fold_idx+1}/5"})

        (
            _,
            fold_test_idx,
            fold_preds,
            fold_train_idx,
            fold_train_preds,
        ) = train_single_fold_linear_nt(
            fold_idx,
            safe_Y,
            gene_emb,
            tf_emb,
            train_indices,
            test_indices,
            lam=args.lambda_ridge,
        )

        preds[fold_test_idx] = fold_preds
        assign_mask = ~train_filled[fold_train_idx]
        if assign_mask.any():
            train_preds[fold_train_idx[assign_mask]] = fold_train_preds[assign_mask]
            train_filled[fold_train_idx[assign_mask]] = True

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    if (~train_filled).any():
        missing_idx = np.where(~train_filled)[0]
        train_preds[missing_idx] = safe_Y[missing_idx]
        print(f"[warn] Filled {len(missing_idx)} training rows with imputed targets due to missing predictions.")

    return preds, train_preds


# =========================
# main
# =========================

def main():
    args = parse_args()
    script_start = time.time()

    np.random.seed(args.seed)

    repo_root = resolve_repo_root(args.repo_root)
    results_base = repo_root / "results" / args.model_name
    results_base.mkdir(parents=True, exist_ok=True)

    # auto cv_splits_file path if not provided
    if args.cv_splits_file:
        cv_splits_file = args.cv_splits_file
    else:
        cv_splits_file = str(results_base / f"cv_splits_5fold_{args.seed}.pkl")

    # load expression + gene_index
    load_start = time.time()
    targets_df, gene_index = load_targets_and_genes(repo_root, args)
    load_time = time.time() - load_start

    pert_ids = targets_df.index.to_list()
    genes = gene_index
    P, G = targets_df.shape

    print(f"[info] Loaded targets: {P} perturbations × {G} genes.")

    # load NT embeddings and align
    gene_emb, tf_emb, aligned_perts = load_nt_embeddings(
        repo_root,
        args,
        gene_index=genes,
        perturbations=targets_df.index,
    )

    # 根据 aligned_perts 对 targets_df 子集
    targets_df = targets_df.loc[aligned_perts]
    tf_emb = tf_emb  # already aligned to aligned_perts
    pert_ids = aligned_perts
    P, G = targets_df.shape

    print(f"[info] After aligning TF embeddings: {P} perturbations × {G} genes.")
    print(f"[info] Gene embedding shape: {gene_emb.shape}, TF embedding shape: {tf_emb.shape}")

    Y = targets_df.values.astype(np.float32)

    # run 5-fold linear model with NT embeddings
    train_start = time.time()
    preds, train_preds = five_fold_cv_linear_nt(
        Y,
        gene_emb,
        tf_emb,
        args,
        use_progress=args.progress,
        cv_splits_file=cv_splits_file,
    )
    train_time = time.time() - train_start

    # save predictions
    preds_df = pd.DataFrame(preds, index=pert_ids, columns=genes)
    preds_path = results_base / f"predictions_{args.model_name}.parquet"
    preds_df.to_parquet(preds_path)
    print(f"[info] Saved predictions to {preds_path}")

    # plot pred vs gt
    plot_path = results_base / "pred_vs_gt.png"
    plot_pred_vs_gt(Y, preds, plot_path, title=f"{args.model_name}: Pred vs GT")
    print(f"[info] Saved plot to {plot_path}")

    train_plot_path = results_base / "pred_vs_gt_train.png"
    plot_pred_vs_gt(Y, train_preds, train_plot_path, title=f"{args.model_name}: Train Pred vs GT")
    print(f"[info] Saved train-set plot to {train_plot_path}")

    # evaluate
    pear_per, pear_mean = centered_pearson_per_pert(Y, preds)
    rmse_per, rmse_mean = centered_rmse_per_pert(Y, preds)
    mae_per, mae_mean = centered_mae_per_pert(Y, preds)
    mse_center_per, mse_center_mean = centered_mse_per_pert(Y, preds)
    mse_raw_per, mse_raw_mean = mse_per_pert(Y, preds)
    l2_per, l2_mean = l2_loss_per_pert(Y, preds)

    metrics = {
        "centered_pearson_per_pert": pear_per,
        "centered_pearson_mean": pear_mean,
        "centered_rmse_per_pert": rmse_per,
        "centered_rmse_mean": rmse_mean,
        "centered_mae_per_pert": mae_per,
        "centered_mae_mean": mae_mean,
        "centered_mse_per_pert": mse_center_per,
        "centered_mse_mean": mse_center_mean,
        "raw_mse_per_pert": mse_raw_per,
        "raw_mse_mean": mse_raw_mean,
        "l2_loss_per_pert": l2_per,
        "l2_loss_mean": l2_mean,
    }

    metrics_path = results_base / "metrics_linear_nt_5fold.json"
    save_metrics_json(metrics, metrics_path)
    print(f"[info] Saved metrics to {metrics_path}")

    # bootstrap CI for mean centered pearson
    def mean_centered_pear(y_t, y_p):
        _, m = centered_pearson_per_pert(y_t, y_p)
        return m

    obs, (ci_l, ci_u) = paired_bootstrap_ci(
        mean_centered_pear, Y, preds, n_boot=args.bootstrap, random_state=args.seed
    )
    ci_summary = {
        "pearson_mean": obs,
        "pearson_mean_ci_lower": ci_l,
        "pearson_mean_ci_upper": ci_u,
    }
    with open(results_base / "ci_summary_linear_nt_5fold.json", "w") as f:
        json.dump(ci_summary, f, indent=2)

    # per-pert metrics table
    metrics_df = pd.DataFrame(
        {
            "centered_pearson": pear_per,
            "centered_rmse": rmse_per,
            "centered_mae": mae_per,
            "centered_mse": mse_center_per,
            "raw_mse": mse_raw_per,
            "l2_loss": l2_per,
        },
        index=pert_ids,
    )
    metrics_df.to_csv(results_base / "metrics_per_perturbation.csv")

    total_time = time.time() - script_start
    summary = {
        "model_name": args.model_name,
        "model_type": "NT-embedding linear model",
        "cv_method": "5fold",
        "cv_splits_file": cv_splits_file,
        "P": int(P),
        "G": int(G),
        "lambda_ridge": float(args.lambda_ridge),
        "metrics": {
            "centered_pearson_mean": float(pear_mean),
            "centered_rmse_mean": float(rmse_mean),
            "centered_mae_mean": float(mae_mean),
            "centered_mse_mean": float(mse_center_mean),
            "raw_mse_mean": float(mse_raw_mean),
            "l2_loss_mean": float(l2_mean),
        },
        "bootstrap": int(args.bootstrap),
        "timings_seconds": {
            "load_artifacts": load_time,
            "training": train_time,
            "total": total_time,
        },
    }
    with open(results_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[info] Finished. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()