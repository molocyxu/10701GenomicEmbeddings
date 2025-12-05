import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pickle


import matplotlib.pyplot as plt

def plot_pred_vs_gt(y_true, y_pred, out_path, title="Pred vs GT"):
    # Flatten arrays
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    # Remove NaNs
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    
    # Keep only pairs within [-1, 1] for both GT and predictions
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

class ThreeLayerMLP(nn.Module):
    """Three-layer MLP for TF embedding to gene expression prediction."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [1024, 512, 256]):
        super(ThreeLayerMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified MLP on NT TF embeddings")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra",
        help="Base directory",
    )
    parser.add_argument(
        "--ntv2-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra/NT-v2-50M",
        help="Directory containing NT-v2 TF embedding npy files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mlp_nt",
        help="Model name for output files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/litianl/NT_ft/crispra/results",
        help="Output directory for results",
    )
    
    # MLP parameters
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512, 256], 
                       help="Hidden layer dimensions")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--n-top-genes", type=int, default=None, help="Select top N genes by mean absolute expression")
    
    # Cross-validation
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--cv-splits-file", type=str, default=None, help="Path to save/load CV splits")
    
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device with error handling."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                test_tensor = torch.tensor([1.0]).cuda()
                device = torch.device("cuda")
                print(f"[info] CUDA available and working. Using GPU.")
            except Exception as e:
                print(f"[warn] CUDA available but not working: {e}")
                print(f"[info] Falling back to CPU.")
                device = torch.device("cpu")
        else:
            print(f"[info] CUDA not available. Using CPU.")
            device = torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                test_tensor = torch.tensor([1.0]).cuda()
                device = torch.device("cuda")
                print(f"[info] Using CUDA GPU.")
            except Exception as e:
                print(f"[error] CUDA requested but not working: {e}")
                print(f"[info] Falling back to CPU.")
                device = torch.device("cpu")
        else:
            print(f"[error] CUDA requested but not available. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
        print(f"[info] Using device: {device}")
    
    return device


def load_data(data_dir: Path, ntv2_dir: Path, n_top_genes: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load targets and NT TF embeddings data."""
    print(f"[info] Loading targets from {data_dir} and NT embeddings from {ntv2_dir}")
    
    # Load targets
    targets_path = data_dir / "data/artifacts/targets_377.csv"
    print(f"[info] Loading targets from {targets_path}")
    # targets_377.csv has gene_name, gene_id, then expression columns
    # We need gene_id as index
    targets_df = pd.read_csv(targets_path)
    targets_df = targets_df.set_index("gene_id")
    
    # Drop non-numeric columns (like gene_name if it's there, though set_index handled gene_id)
    if "gene_name" in targets_df.columns:
        targets_df = targets_df.drop(columns=["gene_name"])
        
    print(f"[info] Loaded targets: {targets_df.shape} (TFs × genes)")
    
    if n_top_genes is not None:
        # Compute mean absolute expression per gene
        # (TFs x genes), so axis=0
        gene_importance = np.abs(targets_df.values).mean(axis=0)
        if len(gene_importance) > n_top_genes:
            top_indices = np.argsort(gene_importance)[-n_top_genes:]
            top_indices = np.sort(top_indices)
            targets_df = targets_df.iloc[:, top_indices]
            print(f"[info] Selected top {n_top_genes} genes by mean absolute expression.")
    
    # Load NT TF embeddings
    emb_npy_path = ntv2_dir / "NTv2_50m_tf_embeddings.npy"
    ids_npy_path = ntv2_dir / "NTv2_50m_tf_ids.npy"
    
    print(f"[info] Loading NT embeddings from {emb_npy_path}")
    embeddings = np.load(emb_npy_path)
    
    print(f"[info] Loading TF ids from {ids_npy_path}")
    tf_ids_raw = np.load(ids_npy_path)
    tf_ids = [
        id_.decode("utf-8") if isinstance(id_, bytes) else str(id_)
        for id_ in tf_ids_raw
    ]
    
    if len(tf_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch between NT tf_ids ({len(tf_ids)}) and embedding rows ({embeddings.shape[0]})"
        )
        
    # Create DataFrame for embeddings
    tf_emb_df = pd.DataFrame(embeddings, index=[tid.upper() for tid in tf_ids])
    print(f"[info] Loaded NT TF embeddings: {tf_emb_df.shape}")
    
    # Normalize target indices for alignment
    original_index = targets_df.index.to_list()
    upper_index = [idx.upper() for idx in original_index]
    targets_df = targets_df.copy()
    targets_df.index = upper_index
    upper_to_original = {idx.upper(): orig for idx, orig in zip(original_index, original_index)}
    
    # Align data - keep only TFs that exist in both datasets
    common_tfs = targets_df.index.intersection(tf_emb_df.index)
    print(f"[info] Found {len(common_tfs)} common TFs between targets and NT embeddings")
    
    if len(common_tfs) == 0:
        raise ValueError("No common TFs found between targets and embeddings!")
    
    # Filter and align
    targets_aligned = targets_df.loc[common_tfs].sort_index()
    tf_emb_aligned = tf_emb_df.loc[common_tfs].sort_index()
    
    # Restore original IDs for downstream reporting
    restored_index = [upper_to_original.get(idx, idx) for idx in targets_aligned.index]
    targets_aligned.index = restored_index
    tf_emb_aligned.index = restored_index
    
    # Ensure same order
    targets_aligned = targets_aligned.sort_index()
    tf_emb_aligned = tf_emb_aligned.sort_index()
    
    print(f"[info] Final aligned data: {targets_aligned.shape[0]} TFs, {targets_aligned.shape[1]} genes, {tf_emb_aligned.shape[1]} embedding dims")
    
    return targets_aligned, tf_emb_aligned


def save_cv_splits(splits_data: dict, filepath: str):
    """Save cross-validation splits to file for reproducibility."""
    with open(filepath, 'wb') as f:
        pickle.dump(splits_data, f)
    print(f"[info] Saved CV splits to {filepath}")


def load_cv_splits(filepath: str) -> dict:
    """Load cross-validation splits from file."""
    with open(filepath, 'rb') as f:
        splits_data = pickle.load(f)
    print(f"[info] Loaded CV splits from {filepath}")
    return splits_data


def train_mlp_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device
) -> ThreeLayerMLP:
    """Train MLP model with early stopping."""
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    # Create model
    model = ThreeLayerMLP(input_dim, output_dim, args.hidden_dims).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=False,
        num_workers=0
    )
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.7)
    
    # Mixed precision setup
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        try:
            scaler = GradScaler()
        except Exception as e:
            print(f"[warn] Mixed precision not available: {e}")
            scaler = None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    outputs = model(batch_X)
                    finite_mask = torch.isfinite(batch_Y)
                    if finite_mask.any():
                        loss = criterion(outputs[finite_mask], batch_Y[finite_mask])
                    else:
                        continue
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                finite_mask = torch.isfinite(batch_Y)
                if finite_mask.any():
                    loss = criterion(outputs[finite_mask], batch_Y[finite_mask])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
        
        # Validation phase (every 5 epochs)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                if scaler is not None:
                    with autocast():
                        val_outputs = model(X_val)
                else:
                    val_outputs = model(X_val)
                
                finite_mask = torch.isfinite(Y_val)
                if finite_mask.any():
                    val_loss = criterion(val_outputs[finite_mask], Y_val[finite_mask]).item()
                else:
                    val_loss = float('inf')
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def run_5fold_cv(
    X: np.ndarray,
    Y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 5-fold cross-validation."""
    
    P, G = Y.shape
    preds = np.full((P, G), np.nan, dtype=np.float32)
    train_preds = np.full((P, G), np.nan, dtype=np.float32)
    train_filled = np.zeros(P, dtype=bool)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle NaN values in targets
    finite_mask = np.isfinite(Y)
    safe_Y = np.zeros_like(Y, dtype=np.float32)
    for g in range(G):
        gene_finite = finite_mask[:, g]
        if gene_finite.any():
            baseline = float(np.nanmean(Y[:, g]))
            safe_Y[:, g] = np.where(gene_finite, Y[:, g], baseline)
        else:
            safe_Y[:, g] = 0.0
    
    # Create or load CV splits
    fold_splits = None
    if args.cv_splits_file and Path(args.cv_splits_file).exists():
        splits_data = load_cv_splits(args.cv_splits_file)
        stored_total = splits_data.get('total_samples')
        if stored_total == P:
            fold_splits = splits_data['fold_splits']
            print(f"[info] Using existing {args.n_folds}-fold splits")
        else:
            print(f"[warn] Stored splits expect {stored_total} samples but current run has {P}. Regenerating.")
    
    if fold_splits is None:
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(X))
        
        # Save splits for reproducibility
        if args.cv_splits_file:
            splits_data = {
                'method': f'{args.n_folds}fold',
                'n_splits': args.n_folds,
                'random_state': args.seed,
                'fold_splits': fold_splits,
                'total_samples': P
            }
            save_cv_splits(splits_data, args.cv_splits_file)
    
    print(f"[info] Running {args.n_folds}-fold cross-validation...")
    
    # Setup progress bar
    if args.progress:
        pbar = tqdm(total=args.n_folds, desc="CV folds", unit="fold")
    
    for fold_idx, (train_indices, test_indices) in enumerate(fold_splits):
        if args.progress:
            pbar.set_postfix({"fold": f"{fold_idx+1}/{args.n_folds}"})
        
        # Prepare data
        X_train = torch.FloatTensor(X_scaled[train_indices])
        Y_train = torch.FloatTensor(safe_Y[train_indices])
        X_test = torch.FloatTensor(X_scaled[test_indices])
        
        # Split training data for validation
        n_train = len(X_train)
        n_val = max(1, n_train // 5)
        indices = torch.randperm(n_train)
        
        X_train_split = X_train[indices[n_val:]].to(device)
        Y_train_split = Y_train[indices[n_val:]].to(device)
        X_val = X_train[indices[:n_val]].to(device)
        Y_val = Y_train[indices[:n_val]].to(device)
        X_test = X_test.to(device)
        
        # Train model
        model = train_mlp_model(X_train_split, Y_train_split, X_val, Y_val, args, device)
        
        # Predict on validation/test split
        model.eval()
        with torch.no_grad():
            if args.mixed_precision and device.type == 'cuda':
                with autocast():
                    test_preds = model(X_test).cpu().numpy()
            else:
                test_preds = model(X_test).cpu().numpy()
        
        # Store predictions
        preds[test_indices] = test_preds.astype(np.float32)

        # Train-set predictions (first time each perturbation appears in train set)
        with torch.no_grad():
            train_inputs = torch.as_tensor(
                X_scaled[train_indices], dtype=torch.float32, device=device
            )
            if args.mixed_precision and device.type == 'cuda':
                with autocast():
                    train_outputs = model(train_inputs).cpu().numpy()
            else:
                train_outputs = model(train_inputs).cpu().numpy()
        assign_mask = ~train_filled[train_indices]
        if assign_mask.any():
            train_preds[train_indices[assign_mask]] = train_outputs[assign_mask].astype(np.float32)
            train_filled[train_indices[assign_mask]] = True
        
        if args.progress:
            pbar.update(1)
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    if args.progress:
        pbar.close()
    
    if (~train_filled).any():
        missing_idx = np.where(~train_filled)[0]
        train_preds[missing_idx] = safe_Y[missing_idx]
        print(f"[warn] Filled {len(missing_idx)} training rows with imputed targets due to missing predictions.")
    
    return preds, train_preds


def calculate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
    """Calculate evaluation metrics."""
    
    def centered_pearson_per_pert(y_true, y_pred):
        correlations = []
        for i in range(y_true.shape[0]):
            y_true_i = y_true[i, :]
            y_pred_i = y_pred[i, :]
            finite_mask = np.isfinite(y_true_i) & np.isfinite(y_pred_i)
            if finite_mask.sum() > 1:
                corr = np.corrcoef(y_true_i[finite_mask], y_pred_i[finite_mask])[0, 1]
                correlations.append(corr if np.isfinite(corr) else 0.0)
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
    
    # Calculate all metrics
    pear_per, pear_mean = centered_pearson_per_pert(Y_true, Y_pred)
    rmse_per, rmse_mean = rmse_per_pert(Y_true, Y_pred)
    mae_per, mae_mean = mae_per_pert(Y_true, Y_pred)
    mse_per, mse_mean = mse_per_pert(Y_true, Y_pred)
    l2_per, l2_mean = l2_loss_per_pert(Y_true, Y_pred)
    
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


def main():
    args = parse_args()
    script_start = time.time()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load data
    load_start = time.time()
    targets_df, tf_emb_df = load_data(Path(args.data_dir), Path(args.ntv2_dir), args.n_top_genes)
    load_time = time.time() - load_start
    
    # Convert to numpy arrays
    X = tf_emb_df.values.astype(np.float32)
    Y = targets_df.values.astype(np.float32)
    pert_ids = targets_df.index.tolist()
    gene_ids = targets_df.columns.tolist()
    
    print(f"[info] Data shape: {X.shape[0]} TFs × {X.shape[1]} embedding dims → {Y.shape[1]} genes")
    print(f"[info] MLP architecture: {X.shape[1]} → {' → '.join(map(str, args.hidden_dims))} → {Y.shape[1]}")
    
    # Setup output paths
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup CV splits file
    if args.cv_splits_file:
        cv_splits_file = args.cv_splits_file
    else:
        cv_splits_file = str(output_dir / f"cv_splits_{args.n_folds}fold_{args.seed}_{len(pert_ids)}.pkl")
    
    # Run cross-validation
    train_start = time.time()
    preds, train_preds = run_5fold_cv(X, Y, args, device)
    train_time = time.time() - train_start
    
    # Calculate metrics
    metrics = calculate_metrics(Y, preds)
    
    # Save predictions
    preds_path = output_dir / f"predictions_{args.model_name}.parquet"
    preds_df = pd.DataFrame(preds, index=pert_ids, columns=gene_ids)
    preds_df.to_parquet(preds_path)
    print(f"[info] Saved predictions to {preds_path}")
    
    # Plot Pred vs GT
    plot_path = output_dir / "pred_vs_gt.png"
    plot_pred_vs_gt(Y, preds, plot_path, title=f"{args.model_name}: Pred vs GT")
    print(f"[info] Saved plot to {plot_path}")

    train_plot_path = output_dir / "pred_vs_gt_train.png"
    plot_pred_vs_gt(Y, train_preds, train_plot_path, title=f"{args.model_name}: Train Pred vs GT")
    print(f"[info] Saved train-set plot to {train_plot_path}")
    
    # Save metrics
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
    metrics_df.to_csv(output_dir / f"metrics_per_perturbation.csv")
    
    # Save summary
    total_time = time.time() - script_start
    summary = {
        "model_name": args.model_name,
        "model_type": "3-layer MLP (NT embeddings)",
        "cv_method": f"{args.n_folds}fold",
        "cv_splits_file": cv_splits_file,
        "n_tfs": int(Y.shape[0]),
        "n_genes": int(Y.shape[1]),
        "embedding_dim": int(X.shape[1]),
        "mean_centered_pearson": float(metrics["pearson_mean"]),
        "mean_rmse": float(metrics["rmse_mean"]),
        "mean_mae": float(metrics["mae_mean"]),
        "mean_mse": float(metrics["mse_mean"]),
        "mean_l2_loss": float(metrics["l2_loss_mean"]),
        "mlp_params": {
            "hidden_dims": args.hidden_dims,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "mixed_precision": args.mixed_precision,
        },
        "device": str(device),
        "timings_seconds": {
            "load_data": load_time,
            "training": train_time,
            "total": total_time,
        },
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("[info] Finished. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
