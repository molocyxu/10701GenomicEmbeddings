import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from tqdm import tqdm
import json
import time
import math
import matplotlib.pyplot as plt

# Configuration
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
HIDDEN_DIMS = [512, 256, 128]
MIXED_PRECISION = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TOP_GENES = None  # Set to an integer (e.g. 200) to select top N genes by mean absolute expression

# Paths
BASE_PATH = "/home/litianl/NT_ft/crispra/NT-v2-50M"
CSV_PATH = "/home/litianl/NT_ft/crispra/data/artifacts/targets_377.csv"
OUTPUT_DIR = "/home/litianl/NT_ft/crispra/results/concat_mlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_pred_vs_gt(y_true, y_pred, out_path, title="Pred vs GT"):
    # Flatten arrays if they are not already 1D or compatible shapes
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    # Remove NaNs
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    
    if len(yt) == 0:
        return

    # Calculate global correlation for the plot title
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

class CompactExpressionDataset(Dataset):
    def __init__(self, tf_indices, gene_indices, targets, all_tf_embs, all_gene_embs):
        """
        tf_indices: List of indices into all_tf_embs (one per sample)
        gene_indices: List of indices into all_gene_embs (one per sample)
        targets: List of target values
        all_tf_embs: Tensor of all TF embeddings
        all_gene_embs: Tensor of all Gene embeddings
        """
        self.tf_indices = torch.as_tensor(tf_indices, dtype=torch.long)
        self.gene_indices = torch.as_tensor(gene_indices, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        
        self.all_tf_embs = torch.as_tensor(all_tf_embs, dtype=torch.float32)
        self.all_gene_embs = torch.as_tensor(all_gene_embs, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        tf_idx = self.tf_indices[idx]
        gene_idx = self.gene_indices[idx]
        
        tf_emb = self.all_tf_embs[tf_idx]
        gene_emb = self.all_gene_embs[gene_idx]
        
        # Concat
        x = torch.cat([tf_emb, gene_emb], dim=0)
        y = self.targets[idx]
        return x, y

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128]):
        super(MLP, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.1)) # Optional
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_mse_loss = 0
    total_mae_loss = 0
    
    mae_criterion = nn.L1Loss()
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            mse = criterion(pred, y)
            mae = mae_criterion(pred, y)
            
            total_mse_loss += mse.item() * x.size(0)
            total_mae_loss += mae.item() * x.size(0)
            
    mse = total_mse_loss / len(loader.dataset)
    
    return mse

def predict_fold(model, loader, device):
    """Run prediction on a loader and return all predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Predicting", leave=False):
            x = x.to(device)
            pred = model(x)
            preds.extend(pred.cpu().numpy())
    return np.array(preds)

def calculate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray):
    """Calculate evaluation metrics matching other scripts."""
    
    def centered_pearson_per_pert(y_true, y_pred):
        correlations = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 1:
                # Explicit centering to match other scripts (np.corrcoef does this internally but for clarity)
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
    
    return {
        "pearson_mean": pear_mean,
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "mse_mean": mse_mean,
        "l2_loss_mean": l2_mean,
        "pearson_per_pert": pear_per
    }

def main():
    start_time = time.time()
    set_seed(SEED)

    # 1. Load Data
    load_start = time.time()
    tf_embs = np.load(os.path.join(BASE_PATH, "NTv2_50m_tf_embeddings.npy"))
    tf_ids = np.load(os.path.join(BASE_PATH, "NTv2_50m_tf_ids.npy"), allow_pickle=True)
    gene_embs = np.load(os.path.join(BASE_PATH, "NTv2_50m_gene_embeddings.npy"))
    gene_ids = np.load(os.path.join(BASE_PATH, "NTv2_50m_gene_ids.npy"), allow_pickle=True)

    # Create Maps
    tf_id_map = {tid: i for i, tid in enumerate(tf_ids)}
    gene_id_map = {gid: i for i, gid in enumerate(gene_ids)}

    df = pd.read_csv(CSV_PATH)
    
    # Identify TFs and Target Genes
    target_gene_cols = [c for c in df.columns if c.startswith("ENSG") and c != "gene_id"]
    
    if N_TOP_GENES is not None:
        # Compute importance
        temp_mat = df[target_gene_cols].values
        importance = np.mean(np.abs(temp_mat), axis=0)
        if len(importance) > N_TOP_GENES:
            top_idx = np.argsort(importance)[-N_TOP_GENES:]
            top_idx = np.sort(top_idx) # Keep original order
            target_gene_cols = [target_gene_cols[i] for i in top_idx]
            print(f"Selected top {N_TOP_GENES} genes by mean absolute expression.")
    
    # Filter TFs
    valid_tfs_indices = [] 
    tf_emb_indices_per_row = [] 
    
    missing_tfs = 0
    for idx, row in df.iterrows():
        tf_id = row['gene_id']
        if tf_id in tf_id_map:
            valid_tfs_indices.append(idx)
            tf_emb_indices_per_row.append(tf_id_map[tf_id])
        else:
            missing_tfs += 1
            
    # Prepare Target Gene Indices
    target_gene_emb_indices = []
    valid_target_cols = []
    
    for col in target_gene_cols:
        if col in gene_id_map:
            target_gene_emb_indices.append(gene_id_map[col])
            valid_target_cols.append(col)
            
    if len(valid_tfs_indices) == 0 or len(valid_target_cols) == 0:
        print("No valid data found. Exiting.")
        return

    # Subset DF
    df_valid = df.iloc[valid_tfs_indices].reset_index(drop=True)
    tf_emb_indices_per_row = np.array(tf_emb_indices_per_row) 
    
    # (N_tfs, N_genes)
    expression_matrix = df_valid[valid_target_cols].values.astype(np.float32)
    load_end = time.time()
    load_data_time = load_end - load_start
    
    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    tf_row_indices = np.arange(len(df_valid))
    
    # Matrix to store predictions: (N_tfs, N_genes)
    full_preds = np.full_like(expression_matrix, np.nan)
    
    training_start = time.time()
    
    scaler = torch.amp.GradScaler('cuda') if MIXED_PRECISION and DEVICE == 'cuda' else None

    for fold, (train_idx, val_idx) in enumerate(kf.split(tf_row_indices)):
        print(f"Fold {fold+1}/5")
        
        # Construct Train Dataset
        t_idxs = tf_emb_indices_per_row[train_idx]
        train_tf_repeated = np.repeat(t_idxs, len(valid_target_cols))
        
        g_idxs = np.array(target_gene_emb_indices)
        train_gene_tiled = np.tile(g_idxs, len(train_idx))
        
        train_targets = expression_matrix[train_idx, :].flatten()
        
        train_dataset = CompactExpressionDataset(
            train_tf_repeated, train_gene_tiled, train_targets,
            tf_embs, gene_embs
        )
        
        # Construct Val Dataset
        v_idxs = tf_emb_indices_per_row[val_idx]
        val_tf_repeated = np.repeat(v_idxs, len(valid_target_cols))
        val_gene_tiled = np.tile(g_idxs, len(val_idx))
        val_targets = expression_matrix[val_idx, :].flatten()
        
        val_dataset = CompactExpressionDataset(
            val_tf_repeated, val_gene_tiled, val_targets,
            tf_embs, gene_embs
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = MLP(input_dim=1024, hidden_dims=HIDDEN_DIMS).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
            val_mse = evaluate(model, val_loader, criterion, DEVICE)
            
            # Early Stopping
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Predict on validation set
        # Validation set in loader is flattened: (N_val_tfs * N_genes,)
        # We need to reshape back to (N_val_tfs, N_genes)
        flat_preds = predict_fold(model, val_loader, DEVICE)
        
        # Reshape: val_idx size is N_val_tfs. Each TF has N_genes targets.
        # The val_dataset was constructed by repeating TF indices for each gene.
        # Order: TF1-G1, TF1-G2... TF1-Gn, TF2-G1...
        reshaped_preds = flat_preds.reshape(len(val_idx), len(valid_target_cols))
        
        # Store in global matrix
        full_preds[val_idx, :] = reshaped_preds
        
    training_end = time.time()
    training_time = training_end - training_start
    total_time = time.time() - start_time
    
    # Save Predictions
    pert_ids = df_valid['gene_id'].tolist()
    preds_df = pd.DataFrame(full_preds, index=pert_ids, columns=valid_target_cols)
    preds_path = os.path.join(OUTPUT_DIR, "predictions_concat_mlp_cv.parquet")
    preds_df.to_parquet(preds_path)
    print(f"[info] Saved predictions to {preds_path}")
    
    # Plot Pred vs GT
    plot_path = os.path.join(OUTPUT_DIR, "pred_vs_gt.png")
    plot_pred_vs_gt(expression_matrix, full_preds, plot_path, title="Concat MLP: Pred vs GT")
    print(f"[info] Saved plot to {plot_path}")
    
    # Calculate Consistent Metrics
    metrics = calculate_metrics(expression_matrix, full_preds)
    
    summary = {
        "model_name": "concat_mlp_cv",
        "model_type": "Concat Gene-TF MLP",
        "cv_method": "5fold",
        "cv_splits_file": None,
        "n_tfs": len(df_valid),
        "n_genes": len(valid_target_cols),
        "embedding_dim": 512,
        "mean_centered_pearson": float(metrics["pearson_mean"]),
        "mean_rmse": float(metrics["rmse_mean"]),
        "mean_mae": float(metrics["mae_mean"]),
        "mean_mse": float(metrics["mse_mean"]),
        "mean_l2_loss": float(metrics["l2_loss_mean"]),
        "mlp_params": {
            "hidden_dims": HIDDEN_DIMS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "mixed_precision": MIXED_PRECISION
        },
        "device": DEVICE,
        "timings_seconds": {
            "load_data": load_data_time,
            "training": training_time,
            "total": total_time
        }
    }
    
    print(json.dumps(summary, indent=2))
    print(f"Job completed at {time.ctime()}")

if __name__ == "__main__":
    main()
