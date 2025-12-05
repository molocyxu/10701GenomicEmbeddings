#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-fold linear model baseline following the paper's formulation.

- Input: gene expression matrix (perturbations × genes), targets_*.csv/parquet
- Mapping: each perturbation is a TF; we map it to a target gene (read-out gene index)
  using tf_sequences.csv (gene_name/gene_id) and gene_index.txt
- Model:
    Y_train: G × N_train  (rows = genes, cols = perturbations)
    PCA on Y_train_centered → G_embed: G × K
    For each perturbation, its embedding is the row in G_embed of its target gene.
    Solve W in:
        Y_center ≈ G_embed W P^T
      with ridge regularization, using the paper's closed form.
    Predict test perturbations:
        Ŷ_test = G_embed W P_test^T + b

- Cross-validation: 5-fold over perturbations (same split structure as MLP 5-fold).
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
    
    # Keep only values within [-1, 1] range for both GT and predictions
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
    parser = argparse.ArgumentParser(description="5-fold linear model baseline (PCA-based)")

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
        "--model-name",
        type=str,
        default="linear_pca",
        help="Subdirectory under results/ for outputs (default: linear_pca)",
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
        "--k-components",
        type=int,
        default=10,
        help="Number of PCA components K for gene embeddings (default: 10, as in paper).",
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
        # You can customise default name here
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
            # Keep them in original order (optional, but good for consistency)
            top_indices = np.sort(top_indices)
            
            # Subset gene_index and targets
            selected_genes = [gene_index[i] for i in top_indices]
            gene_index = selected_genes
            targets = targets.iloc[:, top_indices]
            print(f"[info] Selected top {args.n_top_genes} genes by mean absolute expression.")

    return targets, gene_index


def build_perturbation_gene_index(
    repo_root: Path,
    pert_ids: List[str],
    gene_index: List[str],
    args: argparse.Namespace,
) -> np.ndarray:
    """
    为每个 perturbation 找到其目标 TF 基因在 read-out genes 中的 index。
    返回一个长度为 P 的数组 pert_to_gene_idx，若找不到则为 -1。

    假设:
      - tf_sequences.csv 有 columns: gene_id, gene_name
      - gene_index.txt 与你下游使用的 read-out gene IDs 一致（通常是 gene_id）
    """
    if args.tf_seq_csv:
        tf_seq_path = Path(args.tf_seq_csv).expanduser().resolve()
    else:
        project_root = repo_root.parent if repo_root.name == "data" else repo_root
        tf_seq_path = project_root / "data" / "tf_sequences.csv"

    if not tf_seq_path.exists():
        raise FileNotFoundError(f"TF sequence CSV not found at {tf_seq_path}. Provide --tf-seq-csv.")

    tf_df = pd.read_csv(tf_seq_path)

    # gene_id -> index in gene_index
    gene_id_to_idx = {str(gid).strip(): i for i, gid in enumerate(gene_index)}

    # gene_name -> gene_id
    name_to_id = {}
    for _, row in tf_df.iterrows():
        gid = str(row["gene_id"]).strip()
        gname = str(row["gene_name"]).strip().upper()
        if gid and gname:
            name_to_id[gname] = gid

    pert_to_gene_idx = np.full(len(pert_ids), -1, dtype=int)
    missing = []

    for i, pert in enumerate(pert_ids):
        key = str(pert).strip().upper()
        gid = None

        # 1) perturbation 本身就是 gene_id？
        if key in gene_id_to_idx:
            gid = key
        # 2) 否则按 gene_name -> gene_id
        elif key in name_to_id:
            gid = name_to_id[key]

        if gid is not None and gid in gene_id_to_idx:
            pert_to_gene_idx[i] = gene_id_to_idx[gid]
        else:
            missing.append(pert)

    if missing:
        print(f"[warn] {len(missing)} perturbations could not be mapped to read-out genes (will use mean baseline).")
        print(f"       examples: {missing[:5]}")

    return pert_to_gene_idx


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
# Linear model: one fold
# =========================

def train_single_fold_linear(
    fold_idx: int,
    Y: np.ndarray,                  # (P, G)
    pert_to_gene_idx: np.ndarray,   # (P,)
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    K: int = 10,
    lam: float = 0.1,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在一个 fold 上训练线性模型并预测 test perturbations。

    - Y: (P, G) perturbation × gene  (已填好 NaN)
    - pert_to_gene_idx: 每个 perturbation 对应的 target gene index（-1 表示缺失）
    - train_indices / test_indices: 这个 fold 的划分
    """

    P_total, G = Y.shape
    # Y_train_full: (G, N_train)
    Y_train_full = Y[train_indices, :].T  # genes × perturb

    # 基因均值 b (G × 1)
    b = Y_train_full.mean(axis=1, keepdims=True)

    # 中心化
    Y_center_full = Y_train_full - b  # (G, N_train)

    # PCA on genes: samples = genes, features = perturbations
    # 有时 N_train 或 G 会限制 K 的最大可用值
    K_eff = min(K, Y_center_full.shape[0], Y_center_full.shape[1])
    if K_eff < 1:
        # 退化情形：直接 baseline
        print(f"[warn][fold {fold_idx}] K_eff < 1, using mean baseline.")
        baseline = b.squeeze(1)  # (G,)
        preds_test = np.tile(baseline, (len(test_indices), 1))
        preds_train = np.tile(baseline, (len(train_indices), 1))
        return (
            fold_idx,
            test_indices,
            preds_test.astype(np.float32),
            train_indices,
            preds_train.astype(np.float32),
        )

    pca = PCA(n_components=K_eff, svd_solver="full")
    G_embed = pca.fit_transform(Y_center_full)  # (G, K_eff)

    # 有效的训练 perturbations（有 target gene）
    train_gene_idx = pert_to_gene_idx[train_indices]
    valid_train_mask = train_gene_idx >= 0
    if valid_train_mask.sum() < 2:
        # 几乎没有有效训练样本，fallback baseline
        print(f"[warn][fold {fold_idx}] Too few valid train perturbations, using mean baseline.")
        baseline = b.squeeze(1)
        preds_test = np.tile(baseline, (len(test_indices), 1))
        preds_train = np.tile(baseline, (len(train_indices), 1))
        return (
            fold_idx,
            test_indices,
            preds_test.astype(np.float32),
            train_indices,
            preds_train.astype(np.float32),
        )

    train_valid_indices = train_indices[valid_train_mask]
    train_valid_gene_idx = train_gene_idx[valid_train_mask]

    # 只用 valid train columns 拟合 W
    Y_train_sub = Y_train_full[:, valid_train_mask]        # (G, N_valid)
    Y_train_sub_center = Y_train_sub - b                   # (G, N_valid)

    # P_train: (N_valid, K_eff)
    P_train = G_embed[train_valid_gene_idx, :]             # target genes 的 embedding

    # 构造闭式解中的各项
    GTG = G_embed.T @ G_embed                              # (K_eff, K_eff)
    GTY = G_embed.T @ Y_train_sub_center                   # (K_eff, N_valid)
    PTP = P_train.T @ P_train                              # (K_eff, K_eff)

    lamI_G = lam * np.eye(K_eff, dtype=np.float64)
    lamI_P = lam * np.eye(K_eff, dtype=np.float64)

    # A = (G^T G + λI)^(-1) G^T Y_center
    A = np.linalg.solve(GTG + lamI_G, GTY)                 # (K_eff, N_valid)
    # W_temp = A P_train
    W_temp = A @ P_train                                   # (K_eff, K_eff)
    # W = W_temp (P^T P + λI)^(-1)
    W = W_temp @ np.linalg.inv(PTP + lamI_P)               # (K_eff, K_eff)

    # 对 test 集预测
    preds_test = np.zeros((len(test_indices), G), dtype=np.float64)
    test_gene_idx = pert_to_gene_idx[test_indices]
    valid_test_mask = test_gene_idx >= 0
    invalid_test_mask = ~valid_test_mask

    # 1) valid test perturbations 用 linear model
    if valid_test_mask.any():
        test_valid_indices = test_indices[valid_test_mask]
        test_valid_gene_idx = test_gene_idx[valid_test_mask]

        P_test = G_embed[test_valid_gene_idx, :]           # (N_test_valid, K_eff)
        # Ŷ = G_embed W P_test^T + b   → (G, N_test_valid)
        Y_hat_valid = G_embed @ W @ P_test.T + b           # (G, N_test_valid)
        preds_test[valid_test_mask, :] = Y_hat_valid.T     # (N_test_valid, G)

    # 2) invalid test perturbations 用 mean baseline
    if invalid_test_mask.any():
        baseline = b.squeeze(1)                            # (G,)
        preds_test[invalid_test_mask, :] = np.tile(baseline, (invalid_test_mask.sum(), 1))
    
    # 3) training predictions
    train_preds = np.zeros((len(train_indices), G), dtype=np.float64)
    train_baseline = b.squeeze(1)
    if valid_train_mask.any():
        Y_hat_train = G_embed @ W @ P_train.T + b          # (G, N_valid)
        train_preds[valid_train_mask, :] = Y_hat_train.T
    if (~valid_train_mask).any():
        train_preds[~valid_train_mask, :] = np.tile(
            train_baseline, (np.count_nonzero(~valid_train_mask), 1)
        )

    return (
        fold_idx,
        test_indices,
        preds_test.astype(np.float32),
        train_indices,
        train_preds.astype(np.float32),
    )


# =========================
# 5-fold CV wrapper
# =========================

def five_fold_cv_linear(
    Y: np.ndarray,
    pert_to_gene_idx: np.ndarray,
    args: argparse.Namespace,
    use_progress: bool = False,
    cv_splits_file: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    5-fold CV over perturbations using the PCA-based linear model.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    P, G = Y.shape
    preds = np.full((P, G), np.nan, dtype=np.float32)
    train_preds = np.full((P, G), np.nan, dtype=np.float32)
    train_filled = np.zeros(P, dtype=bool)

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

    print(f"[info] Running 5-fold CV with linear model (K={args.k_components}, λ={args.lambda_ridge})...")

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
        ) = train_single_fold_linear(
            fold_idx,
            Y,
            pert_to_gene_idx,
            train_indices,
            test_indices,
            K=args.k_components,
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
        train_preds[missing_idx] = Y[missing_idx]
        print(f"[warn] Filled {len(missing_idx)} training rows with ground truth due to missing predictions.")

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

    # build mapping: perturbation -> target gene index
    pert_to_gene_idx = build_perturbation_gene_index(repo_root, pert_ids, genes, args)

    Y = targets_df.values.astype(np.float32)

    # run 5-fold linear model
    train_start = time.time()
    preds, train_preds = five_fold_cv_linear(
        Y,
        pert_to_gene_idx,
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

    metrics_path = results_base / "metrics_linear_5fold.json"
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
    with open(results_base / "ci_summary_linear_5fold.json", "w") as f:
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
        "model_type": "PCA-based linear model",
        "cv_method": "5fold",
        "cv_splits_file": cv_splits_file,
        "P": int(P),
        "G": int(G),
        "k_components": int(args.k_components),
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