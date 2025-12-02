"""
Evaluation utilities for perturbation prediction.

Contains functions for Systema-style demeaning, centered Pearson, RMSE,
paired bootstrap CIs, simple gene-set recovery (AUROC/AUPRC), and plotting/saving helpers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Dict, Tuple, List


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def demean_per_perturbation(y: np.ndarray, axis: int = 1) -> np.ndarray:
    """Center each perturbation vector by subtracting its mean across genes.

    Args:
        y: array of shape (P, G) or (P,) depending on axis. Defaults assume P x G.
        axis: axis to compute mean over (default 1: genes axis).
    Returns:
        demeaned array with same shape as y.
    """
    y = np.asarray(y)
    mean = y.mean(axis=axis, keepdims=True)
    return y - mean


def centered_pearson_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute Pearson correlation per perturbation after demeaning.

    Returns (per_pert_array, mean_value)
    """
    y_true_d = demean_per_perturbation(y_true, axis=1)
    y_pred_d = demean_per_perturbation(y_pred, axis=1)

    per = []
    for t, p in zip(y_true_d, y_pred_d):
        # Pearson via covariance / std
        cov = np.cov(t, p, ddof=0)[0, 1]
        denom = (t.std(ddof=0) * p.std(ddof=0))
        if denom == 0:
            # If either vector is constant (zero variance), correlation is undefined;
            # treat as zero correlation for evaluation purposes.
            per.append(0.0)
        else:
            per.append(cov / denom)
    per = np.array(per)
    return per, np.nanmean(per)


def centered_rmse_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    y_true_d = demean_per_perturbation(y_true, axis=1)
    y_pred_d = demean_per_perturbation(y_pred, axis=1)
    per = np.sqrt(np.mean((y_true_d - y_pred_d) ** 2, axis=1))
    return per, float(np.mean(per))


def centered_mae_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    y_true_d = demean_per_perturbation(y_true, axis=1)
    y_pred_d = demean_per_perturbation(y_pred, axis=1)
    per = np.mean(np.abs(y_true_d - y_pred_d), axis=1)
    return per, float(np.mean(per))


def centered_mse_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """Centered mean squared error per perturbation (demeaned vectors)."""
    y_true_d = demean_per_perturbation(y_true, axis=1)
    y_pred_d = demean_per_perturbation(y_pred, axis=1)
    per = np.mean((y_true_d - y_pred_d) ** 2, axis=1)
    return per, float(np.mean(per))


def mse_per_pert(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """Raw mean squared error per perturbation (no demeaning)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per = np.mean((y_true - y_pred) ** 2, axis=1)
    return per, float(np.mean(per))


def paired_bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                        y_true: np.ndarray, y_pred: np.ndarray,
                        n_boot: int = 1000, alpha: float = 0.05, random_state: int = 0) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.RandomState(random_state)
    P = y_true.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, P, size=P)
        val = metric_fn(y_true[idx], y_pred[idx])
        vals.append(val)
    vals = np.array(vals)
    lower = np.percentile(vals, 100 * (alpha / 2))
    upper = np.percentile(vals, 100 * (1 - alpha / 2))
    obs = metric_fn(y_true, y_pred)
    return float(obs), (float(lower), float(upper))


def compute_gene_set_recovery(y_true: np.ndarray, y_pred: np.ndarray, de_binary: np.ndarray) -> Dict[str, float]:
    """Compute AUROC and AUPRC per perturbation for recovering DE genes.

    Args:
        y_true: P x G continuous responses (not used here except to check shapes)
        y_pred: P x G predicted scores
        de_binary: P x G binary matrix (1=DE gene in ground truth)
    Returns:
        dict with mean AUROC and mean AUPRC and arrays per perturbation
    """
    P = y_true.shape[0]
    aurocs = []
    auprcs = []
    for i in range(P):
        y_t = de_binary[i]
        y_s = y_pred[i]
        # require at least one positive and one negative
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            aurocs.append(np.nan)
            auprcs.append(np.nan)
            continue
        try:
            aurocs.append(roc_auc_score(y_t, y_s))
        except Exception:
            aurocs.append(np.nan)
        try:
            auprcs.append(average_precision_score(y_t, y_s))
        except Exception:
            auprcs.append(np.nan)
    return {
        'auroc_per_pert': np.array(aurocs),
        'auprc_per_pert': np.array(auprcs),
        'auroc_mean': float(np.nanmean(aurocs)),
        'auprc_mean': float(np.nanmean(auprcs)),
    }


def save_metrics(metrics: Dict[str, np.ndarray], out_dir: str = 'results', prefix: str = 'metrics') -> None:
    out = Path(out_dir)
    ensure_dir(out)
    # Save per-perturbation arrays as CSV and summary as JSON
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            pd.Series(v).to_csv(out / f"{prefix}_{k}.csv", index=False)
    # Summary
    summary = {k: (float(v) if np.isscalar(v) or isinstance(v, (float, int)) else None) for k, v in metrics.items()}
    with open(out / f"{prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def plot_metric_hist(per_pert_vals: np.ndarray, out_path: str, title: str = '', xlim: Tuple[float, float] = None, bins: int = 30):
    """Plot a histogram of per-perturbation metric values.

    Args:
        per_pert_vals: 1D array of metric values (may contain NaN).
        out_path: output image path.
        title: plot title.
        xlim: optional (xmin, xmax) to set x-axis limits for zooming.
        bins: number of histogram bins.
    """
    outp = Path(out_path)
    ensure_dir(outp.parent)
    vals = per_pert_vals[~np.isnan(per_pert_vals)]
    plt.figure(figsize=(6, 4))
    sns.histplot(vals, kde=False, bins=bins)
    plt.title(title)
    plt.xlabel('Value')
    if xlim is not None:
        try:
            plt.xlim(xlim)
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()


def plot_pred_vs_true_for_pert(y_true: np.ndarray, y_pred: np.ndarray, pert_idx: int, out_path: str, n_points: int = 2000):
    outp = Path(out_path)
    ensure_dir(outp.parent)
    t = y_true[pert_idx]
    p = y_pred[pert_idx]
    idx = np.arange(len(t))
    if len(t) > n_points:
        idx = np.random.choice(idx, size=n_points, replace=False)
    plt.figure(figsize=(5, 5))
    plt.scatter(t[idx], p[idx], s=5, alpha=0.6)
    plt.xlabel('True (demeaned)')
    plt.ylabel('Pred (demeaned)')
    plt.title(f'Perturbation {pert_idx}: pred vs true')
    plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--')
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
