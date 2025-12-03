"""
Utilities for reading our custom numpy-based embeddings format.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def read_tf_embeddings(path):
    """
    Read TF embeddings from .npz format.
    Falls back to .parquet if .npz doesn't exist.
    Returns DataFrame.
    """
    path = Path(path)
    
    # Try .npz first
    npz_path = path.with_suffix('.npz')
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        df = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
        return df
    
    # Fall back to parquet
    if path.exists():
        return pd.read_parquet(path)
    
    raise FileNotFoundError(f"Neither {npz_path} nor {path} found")


def read_targets(path):
    """Read targets from parquet."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Targets file not found: {path}")


def read_covariates(path):
    """Read covariates from parquet."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Covariates file not found: {path}")
