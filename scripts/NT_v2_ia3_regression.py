"""
NT-v2 + IA3 fine-tuning script for TF gene expression regression.
Inputs:
- tf_sequences_12kb_overlap.csv (for sequences)
- targets_377.csv (for labels, regression targets)
- cv_splits_5fold_42.pkl (5-fold splits)
"""

import os
import argparse
import pickle
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.optimization import get_constant_schedule

from peft import IA3Config, TaskType, get_peft_model

@dataclass
class TaskConfig:
    base_ckpt_path: str
    tokenizer_model_id: str
    sequences_csv_path: str
    targets_csv_path: str
    cv_splits_path: str
    max_length: int = 12200
    learning_rate: float = 3e-3
    batch_size: int = 8
    eval_batch_size: int = 64
    num_epochs: int = 10
    max_steps: int = 10000
    output_dir: str = "results/ntv2_ia3_regression"
    num_labels: int = 4914

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    return device

def build_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    print(f"[Info] Loaded tokenizer from `{model_id}`")
    return tokenizer

def load_data(
    sequences_path: str, 
    targets_path: str
) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    print(f"[Info] Loading sequences from {sequences_path}")
    seq_df = pd.read_csv(sequences_path)
    seq_map = dict(zip(seq_df['gene_id'], seq_df['sequence']))
    print(f"[Info] Loading targets from {targets_path}")
    targets_df = pd.read_csv(targets_path)
    target_cols = [c for c in targets_df.columns if c.startswith("ENSG") and c != "gene_id"]
    print(f"[Info] Found {len(target_cols)} target genes.")
    sequences = []
    labels = []
    valid_tfs = []
    missing_count = 0
    for idx, row in targets_df.iterrows():
        tf_id = row['gene_id']
        if tf_id in seq_map:
            sequences.append(seq_map[tf_id])
            labels.append(row[target_cols].values.astype(np.float32))
            valid_tfs.append(tf_id)
        else:
            missing_count += 1
    print(f"[Info] Loaded {len(sequences)} valid TFs. Missing sequences for {missing_count} TFs.")
    return sequences, np.array(labels), valid_tfs, target_cols

def load_cv_splits(filepath: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    with open(filepath, "rb") as f:
        splits_data = pickle.load(f)
    print(f"[Info] Loaded CV splits from {filepath}")
    return splits_data['fold_splits']

def tokenize_datasets(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["sequence"],
        )
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sequence"],
    )
    print(f"[Info] Tokenized dataset with {len(tokenized)} examples.")
    return tokenized

class NTIa3Trainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = get_constant_schedule(optimizer)
        return self.lr_scheduler

def build_ia3_model(base_ckpt_path: str, num_labels: int) -> torch.nn.Module:
    print(f"[Info] Loading base model from `{base_ckpt_path}`")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_ckpt_path,
        num_labels=num_labels,
        problem_type="regression", 
        trust_remote_code=True,
    )
    ia3_modules = [
        "attention.self.key",
        "attention.self.value",
        "intermediate.dense",
        "output.dense",
    ]
    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=ia3_modules,
        feedforward_modules=["intermediate.dense", "output.dense"],
    )
    ia3_model = get_peft_model(base_model, ia3_config)
    ia3_model.print_trainable_parameters()
    return ia3_model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    pearsons = []
    for i in range(len(labels)):
        if np.std(predictions[i]) > 1e-9 and np.std(labels[i]) > 1e-9:
            res = pearsonr(labels[i], predictions[i])
            pearsons.append(res[0])
        else:
            pearsons.append(0.0)
    mean_pearson = np.mean(pearsons)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": mean_pearson
    }

def calculate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
    def centered_pearson_per_pert(y_true, y_pred):
        correlations = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 1:
                y_true_c = y_true_i[finite_mask] - np.mean(y_true_i[finite_mask])
                y_pred_c = y_pred_i[finite_mask] - np.mean(y_pred_i[finite_mask])
                if np.std(y_true_c) > 1e-9 and np.std(y_pred_c) > 1e-9:
                    corr, _ = pearsonr(y_true_c, y_pred_c)
                    correlations.append(corr if np.isfinite(corr) else 0.0)
                else:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)
        correlations = np.array(correlations)
        return correlations, float(np.nanmean(correlations))
    def centered_rmse_per_pert(y_true, y_pred):
        rmse_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                y_true_c = y_true_i[finite_mask] - np.mean(y_true_i[finite_mask])
                y_pred_c = y_pred_i[finite_mask] - np.mean(y_pred_i[finite_mask])
                rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
                rmse_values.append(rmse)
            else:
                rmse_values.append(float('inf'))
        rmse_values = np.array(rmse_values)
        return rmse_values, float(np.nanmean(rmse_values))
    def centered_mae_per_pert(y_true, y_pred):
        mae_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                y_true_c = y_true_i[finite_mask] - np.mean(y_true_i[finite_mask])
                y_pred_c = y_pred_i[finite_mask] - np.mean(y_pred_i[finite_mask])
                mae = np.mean(np.abs(y_true_c - y_pred_c))
                mae_values.append(mae)
            else:
                mae_values.append(float('inf'))
        mae_values = np.array(mae_values)
        return mae_values, float(np.nanmean(mae_values))
    def centered_mse_per_pert(y_true, y_pred):
        mse_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                y_true_c = y_true_i[finite_mask] - np.mean(y_true_i[finite_mask])
                y_pred_c = y_pred_i[finite_mask] - np.mean(y_pred_i[finite_mask])
                mse = np.mean((y_true_c - y_pred_c)**2)
                mse_values.append(mse)
            else:
                mse_values.append(float('inf'))
        mse_values = np.array(mse_values)
        return mse_values, float(np.nanmean(mse_values))
    def rmse_per_pert(y_true, y_pred):
        rmse_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                rmse = np.sqrt(np.mean((y_true_i[finite_mask] - y_pred_i[finite_mask])**2))
                rmse_values.append(rmse)
            else:
                rmse_values.append(float('inf'))
        rmse_values = np.array(rmse_values)
        return rmse_values, float(np.nanmean(rmse_values))
    def mae_per_pert(y_true, y_pred):
        mae_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                mae = np.mean(np.abs(y_true_i[finite_mask] - y_pred_i[finite_mask]))
                mae_values.append(mae)
            else:
                mae_values.append(float('inf'))
        mae_values = np.array(mae_values)
        return mae_values, float(np.nanmean(mae_values))
    def mse_per_pert(y_true, y_pred):
        mse_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                mse = np.mean((y_true_i[finite_mask] - y_pred_i[finite_mask])**2)
                mse_values.append(mse)
            else:
                mse_values.append(float('inf'))
        mse_values = np.array(mse_values)
        return mse_values, float(np.nanmean(mse_values))
    def l2_loss_per_pert(y_true, y_pred):
        l2_values = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 0:
                l2_loss = np.linalg.norm(y_true_i[finite_mask] - y_pred_i[finite_mask], ord=2)
                l2_values.append(l2_loss)
            else:
                l2_values.append(float('inf'))
        l2_values = np.array(l2_values)
        return l2_values, float(np.nanmean(l2_values))
    pear_per, pear_mean = centered_pearson_per_pert(Y_true, Y_pred)
    rmse_per, rmse_mean = rmse_per_pert(Y_true, Y_pred)
    mae_per, mae_mean = mae_per_pert(Y_true, Y_pred)
    mse_per, mse_mean = mse_per_pert(Y_true, Y_pred)
    l2_per, l2_mean = l2_loss_per_pert(Y_true, Y_pred)
    rmse_c_per, rmse_c_mean = centered_rmse_per_pert(Y_true, Y_pred)
    mae_c_per, mae_c_mean = centered_mae_per_pert(Y_true, Y_pred)
    mse_c_per, mse_c_mean = centered_mse_per_pert(Y_true, Y_pred)
    return {
        "pearson_mean": pear_mean,
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "mse_mean": mse_mean,
        "l2_loss_mean": l2_mean,
        "centered_rmse_mean": rmse_c_mean,
        "centered_mae_mean": mae_c_mean,
        "centered_mse_mean": mse_c_mean,
        "pearson_per_pert": pear_per,
        "rmse_per_pert": rmse_per,
        "mae_per_pert": mae_per,
        "mse_per_pert": mse_per,
        "l2_loss_per_pert": l2_per,
        "centered_rmse_per_pert": rmse_c_per,
        "centered_mae_per_pert": mae_c_per,
        "centered_mse_per_pert": mse_c_per
    }

def run_finetuning(cfg: TaskConfig):
    device = get_device()
    sequences, labels, valid_tfs, target_cols = load_data(cfg.sequences_csv_path, cfg.targets_csv_path)
    cfg.num_labels = labels.shape[1]
    print(f"[Info] Num labels (target genes): {cfg.num_labels}")
    tokenizer = build_tokenizer(cfg.tokenizer_model_id)
    cv_splits = load_cv_splits(cfg.cv_splits_path)
    fold_results = []
    all_preds = np.zeros_like(labels)
    test_indices_seen = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        print(f"\n[Info] ===== Fold {fold_idx}/{len(cv_splits)} =====")
        X_train = [sequences[i] for i in train_idx]
        y_train = labels[train_idx]
        X_test = [sequences[i] for i in test_idx]
        y_test = labels[test_idx]
        ds_train = Dataset.from_dict({"sequence": X_train, "labels": y_train})
        ds_test = Dataset.from_dict({"sequence": X_test, "labels": y_test})
        tokenized_train = tokenize_datasets(ds_train, tokenizer, cfg.max_length)
        tokenized_test = tokenize_datasets(ds_test, tokenizer, cfg.max_length)
        model = build_ia3_model(cfg.base_ckpt_path, cfg.num_labels).to(device)
        training_args = TrainingArguments(
            output_dir=os.path.join(cfg.output_dir, f"fold_{fold_idx}"),
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="pearson", 
            greater_is_better=True,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="constant",
            warmup_steps=0,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            num_train_epochs=cfg.num_epochs,
            logging_steps=100,
            label_names=["labels"],
            dataloader_drop_last=False,
            report_to="none"
        )
        trainer = NTIa3Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        preds_output = trainer.predict(tokenized_test)
        fold_preds = preds_output.predictions
        all_preds[test_idx] = fold_preds
        test_indices_seen.extend(test_idx)
        metrics = preds_output.metrics
        print(f"[Info] Fold {fold_idx} Test Metrics: {metrics}")
        fold_results.append(metrics)
    final_metrics = calculate_metrics(labels, all_preds)
    print("\n[Info] ===== Cross-validation Summary =====")
    print(f"Average Pearson: {final_metrics['pearson_mean']:.4f}")
    print(f"Average MSE: {final_metrics['mse_mean']:.4f}")
    preds_df = pd.DataFrame(all_preds, index=valid_tfs, columns=target_cols)
    os.makedirs(cfg.output_dir, exist_ok=True)
    preds_path = os.path.join(cfg.output_dir, "predictions_nt_ia3_cv.parquet")
    preds_df.to_parquet(preds_path)
    print(f"[Info] Saved all predictions to {preds_path}")
    try:
        import matplotlib.pyplot as plt
        yt = labels.flatten()
        yp = all_preds.flatten()
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        corr, _ = pearsonr(yt, yp)
        plt.figure(figsize=(8, 6))
        plt.hexbin(yt, yp, gridsize=50, cmap='Blues', mincnt=1, bins='log')
        plt.colorbar(label='log10(count)')
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        plt.xlabel('Ground Truth Expression')
        plt.ylabel('Predicted Expression')
        plt.title(f"NT-v2 IA3 Finetune: Pred vs GT\nGlobal Pearson r = {corr:.4f}")
        plt.tight_layout()
        plot_path = os.path.join(cfg.output_dir, "pred_vs_gt.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[Info] Saved plot to {plot_path}")
    except ImportError:
        print("[Warn] matplotlib not found, skipping plot.")
    summary = {
        "model_name": "nt_ia3_finetune",
        "cv_method": "5fold",
        "mean_centered_pearson": float(final_metrics['pearson_mean']),
        "mean_rmse": float(final_metrics['rmse_mean']),
        "mean_mae": float(final_metrics['mae_mean']),
        "mean_mse": float(final_metrics['mse_mean']),
        "mean_l2_loss": float(final_metrics['l2_loss_mean']),
        "mean_centered_rmse": float(final_metrics['centered_rmse_mean']),
        "mean_centered_mae": float(final_metrics['centered_mae_mean']),
        "mean_centered_mse": float(final_metrics['centered_mse_mean']),
        "fold_results": fold_results
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    cfg = TaskConfig(
        base_ckpt_path="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        tokenizer_model_id="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        sequences_csv_path="/home/litianl/NT_ft/crispra/data/artifacts/tf_sequences_12kb_overlap.csv",
        targets_csv_path="/home/litianl/NT_ft/crispra/data/artifacts/targets_377.csv",
        cv_splits_path="/home/litianl/NT_ft/crispra/data/results/linear_nt_embed/cv_splits_5fold_42.pkl",
        output_dir="/home/litianl/NT_ft/crispra/results/ntv2_ia3_finetune",
        max_length=12200,
        batch_size=4,
        eval_batch_size=1,
        learning_rate=1e-3,
        num_epochs=20
    )
    run_finetuning(cfg)
