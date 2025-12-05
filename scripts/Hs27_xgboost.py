"""
5-fold XGBoost baseline on HyenaDNA TF embeddings.

This script mirrors the experimental setup used by `Hs_27_mlp_simplified.py`
so both models operate on the same 377 TF × 4912 gene configuration.
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBRegressor


def plot_pred_vs_gt(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    """Scatter (hexbin) plot comparing predictions with ground truth."""
    yt = y_true.flatten()
    yp = y_pred.flatten()

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    range_mask = (yt >= -1.0) & (yt <= 1.0) & (yp >= -1.0) & (yp <= 1.0)
    yt = yt[range_mask]
    yp = yp[range_mask]

    if len(yt) == 0:
        return

    from scipy.stats import pearsonr

    corr, _ = pearsonr(yt, yp)

    plt.figure(figsize=(8, 6))
    plt.hexbin(yt, yp, gridsize=50, cmap="Blues", mincnt=1, bins="log")
    plt.colorbar(label="log10(count)")

    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)

    plt.xlabel("Ground Truth Expression")
    plt.ylabel("Predicted Expression")
    plt.title(f"{title}\nGlobal Pearson r = {corr:.4f}")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost on HyenaDNA TF embeddings")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra",
        help="Base directory containing data/artifacts",
    )
    parser.add_argument(
        "--hyena-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra/HyenaDNA_outputs",
        help="Directory containing HyenaDNA outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgboost_hyena",
        help="Model name for output sub-directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra/results",
        help="Directory to store predictions/metrics",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=None,
        help="Optionally keep top-N genes by mean absolute expression",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--cv-splits-file",
        type=str,
        default=None,
        help="Path to save/load CV splits for reproducibility",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CV splits")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for MultiOutputRegressor (default mirrors sklearn)",
    )
    return parser.parse_args()


def load_data(data_dir: Path, hyena_dir: Path, n_top_genes: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TF expression targets and HyenaDNA TF embeddings with alignment."""
    targets_path = data_dir / "data/artifacts/targets_377.csv"
    embeddings_path = hyena_dir / "hyenadna_tf_embeddings_mean.npy"
    index_path = hyena_dir / "hyenadna_tf_gene_index.txt"

    targets_df = pd.read_csv(targets_path).set_index("gene_id")
    if "gene_name" in targets_df.columns:
        targets_df = targets_df.drop(columns=["gene_name"])

    if n_top_genes is not None:
        gene_importance = np.abs(targets_df.values).mean(axis=0)
        if len(gene_importance) > n_top_genes:
            top_idx = np.argsort(gene_importance)[-n_top_genes:]
            top_idx = np.sort(top_idx)
            targets_df = targets_df.iloc[:, top_idx]

    embeddings = np.load(embeddings_path)
    with open(index_path, "r") as f:
        tf_ids = [line.strip() for line in f if line.strip()]

    if len(tf_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between HyenaDNA TF index ({len(tf_ids)}) and embeddings ({embeddings.shape[0]})"
        )

    tf_emb_df = pd.DataFrame(embeddings, index=tf_ids)

    common = targets_df.index.intersection(tf_emb_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping TFs between targets and embeddings")

    targets_aligned = targets_df.loc[common].sort_index()
    tf_emb_aligned = tf_emb_df.loc[common].sort_index()

    return targets_aligned, tf_emb_aligned


def save_cv_splits(splits: Dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(splits, f)
    print(f"[info] Saved CV splits to {filepath}")


def load_cv_splits(filepath: Path) -> Dict:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"[info] Loaded CV splits from {filepath}")
    return data


def prepare_safe_targets(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-mask, imputed targets, and per-gene baselines."""
    finite_mask = np.isfinite(Y)
    safe_Y = np.zeros_like(Y, dtype=np.float32)
    baselines = np.zeros(Y.shape[1], dtype=np.float32)

    for g in range(Y.shape[1]):
        gene_mask = finite_mask[:, g]
        if gene_mask.any():
            baseline = float(np.nanmean(Y[:, g]))
            baselines[g] = baseline
            safe_Y[:, g] = np.where(gene_mask, Y[:, g], baseline)
        else:
            baselines[g] = 0.0
            safe_Y[:, g] = 0.0

    return safe_Y, finite_mask, baselines


def run_xgb_cv(
    X: np.ndarray,
    Y: np.ndarray,
    args: argparse.Namespace,
    cv_splits_path: Path,
) -> np.ndarray:
    """Train default XGBoost regressors inside a MultiOutputRegressor across folds."""

    P, G = Y.shape
    preds = np.full((P, G), np.nan, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    safe_Y, finite_mask, gene_baselines = prepare_safe_targets(Y)
    sample_weights = finite_mask.any(axis=1).astype(np.float32)

    fold_splits = None
    if cv_splits_path and cv_splits_path.exists():
        splits = load_cv_splits(cv_splits_path)
        stored_total = splits.get("total_samples")
        stored_folds = splits.get("n_splits")
        if stored_total == P and stored_folds == args.n_folds:
            fold_splits = splits["fold_splits"]
            print(f"[info] Using existing {args.n_folds}-fold splits")
        else:
            print("[warn] Stored splits do not match current data; regenerating.")

    if fold_splits is None:
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(X_scaled))
        if cv_splits_path:
            splits = {
                "method": f"{args.n_folds}fold",
                "n_splits": args.n_folds,
                "random_state": args.seed,
                "fold_splits": fold_splits,
                "total_samples": P,
            }
            save_cv_splits(splits, cv_splits_path)

    pbar = tqdm(total=args.n_folds, desc="CV folds", unit="fold") if args.progress else None

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        if pbar is not None:
            pbar.set_postfix({"fold": f"{fold_idx + 1}/{args.n_folds}"})

        train_weights = sample_weights[train_idx]
        effective = float(train_weights.sum())
        if effective <= 0:
            baseline_preds = np.tile(gene_baselines, (len(test_idx), 1))
            preds[test_idx] = baseline_preds.astype(np.float32)
        else:
            base_model = XGBRegressor()
            model = MultiOutputRegressor(base_model, n_jobs=args.n_jobs if args.n_jobs > 0 else None)
            model.fit(X_scaled[train_idx], safe_Y[train_idx], sample_weight=train_weights)
            fold_preds = model.predict(X_scaled[test_idx]).astype(np.float32)
            preds[test_idx] = fold_preds

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return preds


def calculate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
    """Compute per-perturbation metrics to match the MLP script outputs."""

    def centered_pearson_per_pert(y_true, y_pred):
        values = []
        for i in range(y_true.shape[0]):
            yt = y_true[i, :]
            yp = y_pred[i, :]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() > 1:
                corr = np.corrcoef(yt[mask], yp[mask])[0, 1]
                values.append(corr if np.isfinite(corr) else 0.0)
            else:
                values.append(0.0)
        values = np.array(values)
        return values, float(np.nanmean(values))

    def rmse_per_pert(y_true, y_pred):
        values = []
        for i in range(y_true.shape[0]):
            yt = y_true[i, :]
            yp = y_pred[i, :]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2))
                values.append(rmse)
            else:
                values.append(float("inf"))
        values = np.array(values)
        return values, float(np.nanmean(values))

    def mae_per_pert(y_true, y_pred):
        values = []
        for i in range(y_true.shape[0]):
            yt = y_true[i, :]
            yp = y_pred[i, :]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() > 0:
                mae = np.mean(np.abs(yt[mask] - yp[mask]))
                values.append(mae)
            else:
                values.append(float("inf"))
        values = np.array(values)
        return values, float(np.nanmean(values))

    def mse_per_pert(y_true, y_pred):
        values = []
        for i in range(y_true.shape[0]):
            yt = y_true[i, :]
            yp = y_pred[i, :]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() > 0:
                mse = np.mean((yt[mask] - yp[mask]) ** 2)
                values.append(mse)
            else:
                values.append(float("inf"))
        values = np.array(values)
        return values, float(np.nanmean(values))

    def l2_per_pert(y_true, y_pred):
        values = []
        for i in range(y_true.shape[0]):
            yt = y_true[i, :]
            yp = y_pred[i, :]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() > 0:
                l2 = np.linalg.norm(yt[mask] - yp[mask], ord=2)
                values.append(l2)
            else:
                values.append(float("inf"))
        values = np.array(values)
        return values, float(np.nanmean(values))

    pear_per, pear_mean = centered_pearson_per_pert(Y_true, Y_pred)
    rmse_per, rmse_mean = rmse_per_pert(Y_true, Y_pred)
    mae_per, mae_mean = mae_per_pert(Y_true, Y_pred)
    mse_per, mse_mean = mse_per_pert(Y_true, Y_pred)
    l2_per, l2_mean = l2_per_pert(Y_true, Y_pred)

    return {
        "pearson_per_pert": pear_per,
        "pearson_mean": pear_mean,
        "rmse_per_pert": rmse_per,
        "rmse_mean": rmse_mean,
        "mae_per_pert": mae_per,
        "mae_mean": mae_mean,
        "mse_per_pert": mse_per,
        "mse_mean": mse_mean,
        "l2_loss_per_pert": l2_per,
        "l2_loss_mean": l2_mean,
    }


def main() -> None:
    args = parse_args()
    script_start = time.time()

    np.random.seed(args.seed)

    targets_df, tf_emb_df = load_data(Path(args.data_dir), Path(args.hyena_dir), args.n_top_genes)
    X = tf_emb_df.values.astype(np.float32)
    Y = targets_df.values.astype(np.float32)

    pert_ids = targets_df.index.tolist()
    gene_ids = targets_df.columns.tolist()

    print(f"[info] Data shape: {X.shape[0]} TFs × {X.shape[1]} dims -> {Y.shape[1]} genes")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cv_splits_file:
        cv_splits_path = Path(args.cv_splits_file)
    else:
        cv_splits_path = output_dir / f"cv_splits_{args.n_folds}fold_{args.seed}_{len(pert_ids)}.pkl"

    train_start = time.time()
    preds = run_xgb_cv(X, Y, args, cv_splits_path)
    train_time = time.time() - train_start

    metrics = calculate_metrics(Y, preds)

    preds_path = output_dir / f"predictions_{args.model_name}.parquet"
    pd.DataFrame(preds, index=pert_ids, columns=gene_ids).to_parquet(preds_path)
    print(f"[info] Saved predictions to {preds_path}")

    plot_path = output_dir / "pred_vs_gt.png"
    plot_pred_vs_gt(Y, preds, plot_path, title=f"{args.model_name}: Pred vs GT")

    metrics_df = pd.DataFrame(
        {
            "centered_pearson": metrics["pearson_per_pert"],
            "rmse": metrics["rmse_per_pert"],
            "mae": metrics["mae_per_pert"],
            "mse": metrics["mse_per_pert"],
            "l2_loss": metrics["l2_loss_per_pert"],
        },
        index=pert_ids,
    )
    metrics_df.to_csv(output_dir / "metrics_per_perturbation.csv")

    summary = {
        "model_name": args.model_name,
        "model_type": "XGBoost (default params)",
        "cv_method": f"{args.n_folds}fold",
        "cv_splits_file": str(cv_splits_path),
        "n_tfs": int(Y.shape[0]),
        "n_genes": int(Y.shape[1]),
        "embedding_dim": int(X.shape[1]),
        "mean_centered_pearson": float(metrics["pearson_mean"]),
        "mean_rmse": float(metrics["rmse_mean"]),
        "mean_mae": float(metrics["mae_mean"]),
        "mean_mse": float(metrics["mse_mean"]),
        "mean_l2_loss": float(metrics["l2_loss_mean"]),
        "multioutput_n_jobs": args.n_jobs,
        "timings_seconds": {
            "training": train_time,
            "total": time.time() - script_start,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[info] Finished. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

