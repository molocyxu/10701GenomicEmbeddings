"""
Extract and map TF embeddings to perturbations.

This script loads NT-v2-50m embeddings and maps them to perturbation IDs
based on the tf_sequences.csv and targets.parquet index.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_tf_embeddings_and_ids(nt_emb_dir='data/NT-v2-50m'):
    """Load TF embeddings and their corresponding gene IDs."""
    logger = logging.getLogger(__name__)
    
    nt_dir = Path(nt_emb_dir)
    tf_emb_file = nt_dir / 'NTv2_50m_tf_embeddings.npy'
    tf_ids_file = nt_dir / 'NTv2_50m_tf_ids.npy'
    
    if not tf_emb_file.exists() or not tf_ids_file.exists():
        raise FileNotFoundError(f"TF embeddings not found in {nt_emb_dir}")
    
    tf_embeddings = np.load(str(tf_emb_file))  # shape (n_tf, 512)
    tf_ids = np.load(str(tf_ids_file), allow_pickle=True)  # shape (n_tf,)
    
    logger.info(f"Loaded {len(tf_ids)} TF embeddings (shape {tf_embeddings.shape})")
    
    return tf_embeddings, tf_ids


def load_tf_sequences(tf_seq_csv='data/tf_sequences.csv'):
    """Load TF sequence metadata."""
    logger = logging.getLogger(__name__)
    
    df = pd.read_csv(tf_seq_csv)
    logger.info(f"Loaded {len(df)} TF sequences from {tf_seq_csv}")
    
    return df


def load_targets(targets_path='artifacts/targets.parquet'):
    """Load targets to get perturbation IDs."""
    Y = pd.read_parquet(targets_path)
    return Y


def map_tf_sequences_to_perturbations(tf_seq_df, targets_index):
    """
    Build mapping from perturbation ID (in targets) to TF gene_id.
    
    Strategy: assume perturbation_id in targets matches TF gene_name or gene_id from tf_sequences.csv.
    Falls back to fuzzy matching if needed.
    """
    logger = logging.getLogger(__name__)
    
    # Convert targets index to list
    pert_ids = targets_index.tolist()
    
    # Create reverse lookup: gene_name -> gene_id and gene_id -> gene_id
    name_to_id = {}
    id_to_id = {}
    for _, row in tf_seq_df.iterrows():
        name = str(row['gene_name']).upper().strip()
        gene_id = str(row['gene_id']).strip()
        name_to_id[name] = gene_id
        id_to_id[gene_id] = gene_id
    
    pert_to_tf_id = {}
    matched = 0
    unmatched = []
    
    for pert_id in pert_ids:
        pert_upper = str(pert_id).upper().strip()
        
        # Try direct match first (gene_id)
        if pert_upper in id_to_id:
            pert_to_tf_id[pert_id] = id_to_id[pert_upper]
            matched += 1
        # Try gene_name match
        elif pert_upper in name_to_id:
            pert_to_tf_id[pert_id] = name_to_id[pert_upper]
            matched += 1
        else:
            unmatched.append(pert_id)
    
    logger.info(f"Mapped {matched}/{len(pert_ids)} perturbations to TF gene IDs")
    if unmatched:
        logger.warning(f"Could not map {len(unmatched)} perturbations: {unmatched[:5]}...")
    
    return pert_to_tf_id


def build_tf_embedding_matrix(pert_to_tf_id, tf_ids, tf_embeddings, targets_index):
    """
    Build DataFrame with TF embeddings indexed by perturbation_id.
    
    For perturbations without TF embeddings, use NaN rows.
    """
    logger = logging.getLogger(__name__)
    
    # Create mapping: tf_id -> embedding
    tf_id_to_emb = {str(tf_id): emb for tf_id, emb in zip(tf_ids, tf_embeddings)}
    
    emb_dim = tf_embeddings.shape[1]
    emb_data = []
    
    for pert_id in targets_index:
        if pert_id in pert_to_tf_id:
            tf_id = pert_to_tf_id[pert_id]
            if tf_id in tf_id_to_emb:
                emb_data.append(tf_id_to_emb[tf_id])
            else:
                emb_data.append(np.full(emb_dim, np.nan))
        else:
            emb_data.append(np.full(emb_dim, np.nan))
    
    emb_df = pd.DataFrame(np.vstack(emb_data), index=targets_index)
    emb_df.columns = [f'emb_{i}' for i in range(emb_dim)]
    
    n_valid = (~emb_df.isna().any(axis=1)).sum()
    logger.info(f"TF embedding matrix: {emb_df.shape}, {n_valid} valid embeddings")
    
    return emb_df


def main(targets_path='artifacts/targets.parquet', 
         tf_seq_csv='data/tf_sequences.csv', 
         nt_emb_dir='data/NT-v2-50m',
         out_path='artifacts/tf_embeddings.parquet'):
    """Main extraction pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== TF Embeddings Extraction ===")
    
    logger.info("Loading targets")
    Y = load_targets(targets_path)
    
    logger.info("Loading TF embeddings and IDs")
    tf_embeddings, tf_ids = load_tf_embeddings_and_ids(nt_emb_dir)
    
    logger.info("Loading TF sequences")
    tf_seq_df = load_tf_sequences(tf_seq_csv)
    
    logger.info("Mapping TF sequences to perturbations")
    pert_to_tf_id = map_tf_sequences_to_perturbations(tf_seq_df, Y.index)
    
    logger.info("Building TF embedding matrix")
    emb_df = build_tf_embedding_matrix(pert_to_tf_id, tf_ids, tf_embeddings, Y.index)
    
    logger.info(f"Saving to {out_path}")
    # Save as .npz (numpy compressed) to avoid parquet corruption issues
    if out_path.endswith('.parquet'):
        npz_path = out_path.replace('.parquet', '.npz')
        np.savez_compressed(npz_path, data=emb_df.values, index=emb_df.index.values, columns=emb_df.columns.values)
        logger.info(f"Saved as numpy: {npz_path}")
    else:
        np.savez_compressed(out_path, data=emb_df.values, index=emb_df.index.values, columns=emb_df.columns.values)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
