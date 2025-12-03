"""
Load genes and map them to enformer embeddings.

"""
from pathlib import Path
import sys
import logging
import numpy as np
import pandas as pd
import torch
from enformer_pytorch import Enformer, str_to_one_hot


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

def load_genes(gene_seq_csv):
    """Load gene data."""
    logger = logging.getLogger(__name__)
    
    df = pd.read_csv(gene_seq_csv)
    logger.info(f"Loaded {len(df)} Gene sequences from {gene_seq_csv}")
    
    return df

def main(gene_path=ROOT/'data/Hs27_gene_sequences_12kb_4912genes.csv'):
    """Main extraction pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Enformer Embeddings ===")  
    
    logger.info("Loading gene data")
    genes = load_genes(gene_path)
    encoded = str_to_one_hot(list(genes['sequence']))
    logger.info(encoded.shape)
    logger.info(encoded[:10].shape)

    model = Enformer.from_hparams(
        target_length = 94, # reduced since our 12k seqs don't allow bigger targets
    )

    filename = f"artifacts/enformer-embeddings.pt"
    out_path = Path(ROOT/filename)

    logger.info("Running the model...")
    embeddings = model(encoded[50:60], return_only_embeddings = True)
    torch.save(embeddings,out_path)
    logger.info("Done!")


if __name__ == '__main__':
    main()
