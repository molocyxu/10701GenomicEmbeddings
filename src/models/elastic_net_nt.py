"""Elastic Net M1 variant using only NT TF embeddings.

This script treats perturbations (TF activations) as samples and fits, for each gene,
an ElasticNet model that maps TF embeddings -> gene response. It runs Leave-One-TF-Out
(LOTO) evaluation: for each held-out perturbation, train on the rest and predict the held-out.

Outputs saved under `results/m1_nt/loto/`:
 - `predictions.parquet` : predicted matrix (perturbation x genes)
 - `metrics_per_perturbation.csv` : per-perturbation centered Pearson & RMSE
 - `summary.json` : aggregate means

Quick-test: run with `--max-genes 200` to run faster during development.
"""
import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# make sure `src` is importable (match baseline scripts)
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from evaluation import evaluation as evalmod
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error


def centered_pearson(x, y):
    # x, y are 1D arrays
    xm = x - np.nanmean(x)
    ym = y - np.nanmean(y)
    num = np.nansum(xm * ym)
    den = np.sqrt(np.nansum(xm * xm) * np.nansum(ym * ym))
    if den == 0:
        return 0.0
    return float(num / den)


def fit_predict_gene(X_train, y_train, X_test, alphas=None, l1_ratio=[0.1, 0.5, 0.9, 1.0], max_iter=5000):
    # if y_train is constant, return that constant
    if np.all(np.isnan(y_train)):
        return np.nan
    if np.nanstd(y_train) == 0:
        return float(np.nanmean(y_train))
    # ElasticNetCV requires finite array
    mask = ~np.isnan(y_train)
    if mask.sum() < 5:
        # not enough samples
        return float(np.nanmean(y_train))
    try:
        model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=5, max_iter=max_iter, n_jobs=1)
        model.fit(X_train[mask], y_train[mask])
        pred = model.predict(X_test.reshape(1, -1))[0]
        return float(pred)
    except Exception:
        return float(np.nanmean(y_train))


def main(max_genes=None, n_jobs=4):
    out_dir = 'results/m1_nt/loto'
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    tf_emb = pd.read_parquet('artifacts/tf_embeddings.parquet')
    targets = pd.read_parquet('artifacts/targets.parquet')

    # Align
    common_idx = tf_emb.index.intersection(targets.index)
    if len(common_idx) != len(targets.index):
        print(f"Warning: {len(targets.index)-len(common_idx)} perturbations missing TF embeddings; they will be skipped.")
    tf_emb = tf_emb.loc[common_idx]
    targets = targets.loc[common_idx]

    P, G = targets.shape
    genes = list(targets.columns)
    perturbs = list(targets.index)

    if max_genes is not None:
        genes = genes[:max_genes]
        G = len(genes)

    X = tf_emb.values.astype(float)  # shape (P, D)
    Y = targets[genes].values.astype(float)  # shape (P, G)

    alphas = np.logspace(-4, 2, 20)

    # Prepare storage
    preds = np.zeros_like(Y)
    preds[:] = np.nan

    def process_holdout(i_holdout):
        # Train on all except i_holdout
        mask = np.ones(P, dtype=bool)
        mask[i_holdout] = False
        X_train = X[mask]
        X_test = X[i_holdout]
        Y_train = Y[mask]

        # For each gene, fit and predict
        results = []
        for j in range(G):
            yj = Y_train[:, j]
            pred = fit_predict_gene(X_train, yj, X_test, alphas=alphas)
            results.append(pred)
        return np.array(results)

    print(f"Running LOTO on {P} perturbations, {G} genes, embedding dim {X.shape[1]}")
    # parallel over holdouts
    outputs = Parallel(n_jobs=min(n_jobs, P))(delayed(process_holdout)(i) for i in range(P))
    for i in range(P):
        preds[i, :] = outputs[i]

    preds_df = pd.DataFrame(preds, index=perturbs, columns=genes)
    preds_df.to_parquet(os.path.join(out_dir, 'predictions.parquet'))

    # Metrics per perturbation
    metrics = []
    for i in range(P):
        y_true = Y[i, :]
        y_pred = preds[i, :]
        # center per-perturbation
        y_true_c = y_true - np.nanmean(y_true)
        y_pred_c = y_pred - np.nanmean(y_pred)
        pear = centered_pearson(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics.append({'perturbation': perturbs[i], 'centered_pearson': pear, 'rmse': rmse})

    metrics_df = pd.DataFrame(metrics).set_index('perturbation')
    metrics_df.to_csv(os.path.join(out_dir, 'metrics_per_perturbation.csv'))

    # Use evaluation utilities to compute per-perturbation arrays and plots
    Y_arr = Y  # P x G
    preds_arr = preds  # P x G

    pear_per, pear_mean = evalmod.centered_pearson_per_pert(Y_arr, preds_arr)
    raw_mse_per, raw_mse_mean = evalmod.mse_per_pert(Y_arr, preds_arr)
    rmse_per, rmse_mean = evalmod.centered_rmse_per_pert(Y_arr, preds_arr)

    # Save arrays via evalmod.save_metrics (also write explicit CSVs already done)
    # Plot histograms similar to other baselines
    evalmod.plot_metric_hist(pear_per, os.path.join(out_dir, 'pearson_hist_loto.png'), title='M1_NT: centered Pearson (LOTO)')
    evalmod.plot_metric_hist(raw_mse_per, os.path.join(out_dir, 'raw_mse_hist_loto.png'), title='M1_NT: raw MSE (LOTO)')
    evalmod.plot_metric_hist(rmse_per, os.path.join(out_dir, 'rmse_hist_loto.png'), title='M1_NT: centered RMSE (LOTO)')

    # Plot pred vs true for a few representative perturbations: first, median by pearson, best
    valid_idx = np.where(~np.isnan(pear_per))[0]
    if len(valid_idx) > 0:
        first_idx = valid_idx[0]
        median_idx = valid_idx[len(valid_idx) // 2]
        best_idx = valid_idx[int(np.nanargmax(pear_per[valid_idx]))]
        evalmod.plot_pred_vs_true_for_pert(Y_arr, preds_arr, first_idx, os.path.join(out_dir, f'pred_vs_true_pert_{first_idx}.png'))
        evalmod.plot_pred_vs_true_for_pert(Y_arr, preds_arr, median_idx, os.path.join(out_dir, f'pred_vs_true_pert_{median_idx}.png'))
        evalmod.plot_pred_vs_true_for_pert(Y_arr, preds_arr, best_idx, os.path.join(out_dir, f'pred_vs_true_pert_{best_idx}.png'))

    summary = {
        'mean_centered_pearson': float(pear_mean),
        'mean_raw_mse': float(raw_mse_mean),
        'mean_centered_rmse': float(rmse_mean),
        'P': int(P),
        'G': int(G)
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Done. Summary:')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-genes', type=int, default=None, help='Limit number of genes for a fast smoke test')
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    main(max_genes=args.max_genes, n_jobs=args.n_jobs)
