import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tqdm

def load_model_and_tokenizer(model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"):
    """Generic helper to load a Hugging Face model/tokenizer pair."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_CACHE", None),
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_CACHE", None),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    return model, tokenizer, device


def load_hyenadna_model_and_tokenizer(args):
    """Load HyenaDNA model/tokenizer from the local submodule."""
    hyena_root = (
        Path(args.hyena_repo_dir).expanduser().resolve()
        if args.hyena_repo_dir
        else Path(__file__).resolve().parent / "HyenaDNA" / "hyena-dna"
    )
    if not hyena_root.exists():
        raise FileNotFoundError(
            f"HyenaDNA checkout not found at {hyena_root}. Clone the repo or update --hyena-repo-dir."
        )

    if str(hyena_root) not in sys.path:
        sys.path.insert(0, str(hyena_root))

    try:
        from huggingface import HyenaDNAPreTrainedModel  # type: ignore
        from standalone_hyenadna import CharacterTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Unable to import HyenaDNA helpers. Ensure the hyena-dna repo dependencies are installed."
        ) from exc

    checkpoints_dir = hyena_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_name = args.hyena_model_name.split("/")[-1]
    model = HyenaDNAPreTrainedModel.from_pretrained(
        path=str(checkpoints_dir),
        model_name=pretrained_name,
        download=True,
        config=None,
        device=device.type,
        use_head=False,
        n_classes=2,
    )
    model.to(device)
    model.eval()

    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=args.hyena_max_length + 2,
        padding_side="left",
    )

    print(f"HyenaDNA model loaded on device: {device}")
    return model, tokenizer, device

def load_sequences(csv_path, sequence_col='sequence', id_col='gene_id'):
    """Load sequences from CSV file"""
    print(f"Loading sequences from: {csv_path}")
    df = pd.read_csv(csv_path)
    sequences = df[sequence_col].tolist()
    gene_ids = df[id_col].tolist()
    print(f"Loaded {len(sequences)} sequences")
    return sequences, gene_ids

def tokenize_sequences(sequences, tokenizer, max_length):
    """Tokenize sequences for model input"""
    print("Tokenizing sequences...")
    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return tokenized

def extract_embeddings_batch(
    model,
    tokenizer,
    sequences,
    device,
    batch_size=8,
    pooling_method="mean",
    max_length=2048,
    use_attention_mask=True,
    request_hidden_states=True,
):
    """
    Extract embeddings from sequences in batches

    Args:
        model: The NT-v2 model
        tokenizer: The tokenizer
        sequences: List of DNA sequences
        device: torch device
        batch_size: Batch size for processing
        pooling_method: "cls" for CLS token embedding, "mean" for mean pooling (default)
    """
    all_embeddings = []

    for i in tqdm.tqdm(range(0, len(sequences), batch_size), desc=f"Extracting {pooling_method} embeddings"):
        batch_sequences = sequences[i:i+batch_size]

        # Tokenize batch
        tokenized = tokenize_sequences(batch_sequences, tokenizer, max_length=max_length)

        # Move to device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        with torch.no_grad():
            if use_attention_mask:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=request_hidden_states,
                )
            else:
                outputs = model(input_ids=input_ids)

            if request_hidden_states:
                # Hugging Face models expose hidden_states list
                last_hidden_states = outputs.hidden_states[-1]
            else:
                # HyenaDNA returns the embeddings directly
                last_hidden_states = outputs

            # Ensure the tensor is valid
            if torch.isnan(last_hidden_states).any() or torch.isinf(last_hidden_states).any():
                print(f"Warning: Invalid values in hidden states for batch {i}")
                continue

            if pooling_method == "cls":
                # Use CLS token embedding (always at position 0 for ESM models)
                # All our sequences are 12000 bp, so CLS token should always exist
                if last_hidden_states.size(1) > 0:  # Check if sequence length > 0
                    pooled_embeddings = last_hidden_states[:, 0, :]
                else:
                    print(f"Warning: Empty sequence in batch {i}")
                    continue
            elif pooling_method == "mean":
                # Mean pool across sequence length (excluding padding tokens)
                # Expand attention_mask to match hidden size
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

                # Sum embeddings weighted by attention mask
                sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, dim=1)

                # Sum attention mask for each sequence
                sum_mask = torch.sum(attention_mask_expanded, dim=1)

                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)

                # Mean pooling
                pooled_embeddings = sum_embeddings / sum_mask
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}. Use 'cls' or 'mean'.")

            # Ensure CUDA operations are synchronized
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            all_embeddings.append(pooled_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

def save_embeddings(embeddings, gene_ids, output_file):
    """Save embeddings and gene IDs"""
    print(f"Saving embeddings to: {output_file}")
    np.save(output_file, embeddings)
    print(f"Saved {len(embeddings)} embeddings with shape: {embeddings.shape}")

    # Also save gene IDs for reference
    gene_ids_file = output_file.replace('.npy', '_gene_ids.npy')
    np.save(gene_ids_file, np.array(gene_ids))
    print(f"Saved gene IDs to: {gene_ids_file}")


def save_gene_index(gene_ids, output_file):
    """Persist gene IDs order to a text file."""
    print(f"Saving gene index to: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(map(str, gene_ids)))
    print(f"Wrote {len(gene_ids)} gene IDs to {output_file}")


def run_ntv2_pipeline(model, tokenizer, device, args):
    """Existing NT-v2 workflow that extracts gene and TF embeddings."""
    print(f"Using pooling method: {args.pooling_method}")
    print(f"Batch size: {args.batch_size}")

    gene_csv = "/home/litianl/NT_ft/crispra/data/Hs27_gene_sequences_12kb_4912genes.csv"
    tf_csv = "/home/litianl/NT_ft/crispra/data/tf_sequences.csv"

    print("\n=== Processing Gene Sequences ===")
    gene_sequences, gene_ids = load_sequences(gene_csv)
    gene_embeddings = extract_embeddings_batch(
        model,
        tokenizer,
        gene_sequences,
        device,
        batch_size=args.batch_size,
        pooling_method=args.pooling_method,
        max_length=args.max_seq_length,
    )
    gene_output_file = os.path.join(
        args.output_dir,
        f"NTv2_500m_gene_embeddings_{args.pooling_method}.npy",
    )
    save_embeddings(gene_embeddings, gene_ids, gene_output_file)

    print("\n=== Processing TF Sequences ===")
    tf_sequences, tf_ids = load_sequences(tf_csv)
    tf_embeddings = extract_embeddings_batch(
        model,
        tokenizer,
        tf_sequences,
        device,
        batch_size=args.batch_size,
        pooling_method=args.pooling_method,
        max_length=args.max_seq_length,
    )
    tf_output_file = os.path.join(
        args.output_dir,
        f"NTv2_500m_tf_embeddings_{args.pooling_method}.npy",
    )
    save_embeddings(tf_embeddings, tf_ids, tf_output_file)


def run_hyenadna_pipeline(model, tokenizer, device, args):
    """HyenaDNA workflow operating on the TF 1Mb overlap CSV."""
    print(f"Running HyenaDNA pipeline with pooling method: {args.pooling_method}")
    print(f"HyenaDNA batch size: {args.hyena_batch_size}")
    print(f"Sequence source: {args.tf_overlap_csv}")
    print(f"Max tokenized length: {args.hyena_max_length}")

    sequences, gene_ids = load_sequences(
        args.tf_overlap_csv,
        sequence_col="sequence",
        id_col="gene_id",
    )
    embeddings = extract_embeddings_batch(
        model,
        tokenizer,
        sequences,
        device,
        batch_size=args.hyena_batch_size,
        pooling_method=args.pooling_method,
        max_length=args.hyena_max_length,
        use_attention_mask=False,
        request_hidden_states=False,
    )
    hyena_output = os.path.join(
        args.output_dir,
        f"hyenadna_tf_embeddings_{args.pooling_method}.npy",
    )
    save_embeddings(embeddings, gene_ids, hyena_output)

    gene_index_path = (
        args.gene_index_output
        if args.gene_index_output
        else os.path.join(args.output_dir, "hyenadna_tf_gene_index.txt")
    )
    save_gene_index(gene_ids, gene_index_path)

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from DNA sequences using transformer models")
    parser.add_argument("--pooling_method", type=str, choices=["cls", "mean"], default="mean",
                       help="Pooling method: 'cls' for CLS token embedding, 'mean' for mean pooling (default)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument("--output_dir", type=str, default="/home/litianl/NT_ft/crispra/NT-v2-500M",
                       help="Output directory for embeddings")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum tokenized length for NT-v2 runs")
    parser.add_argument("--pipeline", type=str, choices=["ntv2", "hyenadna"], default="ntv2",
                        help="Which embedding pipeline to execute")
    parser.add_argument("--hyena_model_name", type=str, default="LongSafari/hyenadna-large-1m-seqlen",
                        help="HyenaDNA model identifier")
    parser.add_argument("--tf_overlap_csv", type=str,
                        default="/home/litianl/NT_ft/crispra/data/artifacts/tf_sequences_1mb_overlap.csv",
                        help="CSV containing TF sequences for HyenaDNA extraction")
    parser.add_argument("--gene_index_output", type=str, default=None,
                        help="Optional path to save HyenaDNA gene_index order")
    parser.add_argument("--hyena_max_length", type=int, default=1_000_000,
                        help="Maximum tokenized length for HyenaDNA sequences")
    parser.add_argument("--hyena_batch_size", type=int, default=1,
                        help="Batch size for HyenaDNA extraction (default: 1)")
    parser.add_argument("--hyena_repo_dir", type=str, default=None,
                        help="Optional override for the local hyena-dna repo path")

    args = parser.parse_args()

    # Set CUDA_LAUNCH_BLOCKING for better error reporting
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set cache directories to local writable location
    cache_dir = '/home/litianl/.cache/huggingface'
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['HF_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pipeline == "ntv2":
        model, tokenizer, device = load_model_and_tokenizer()
        run_ntv2_pipeline(model, tokenizer, device, args)
    elif args.pipeline == "hyenadna":
        model, tokenizer, device = load_hyenadna_model_and_tokenizer(args)
        run_hyenadna_pipeline(model, tokenizer, device, args)
    else:
        raise ValueError(f"Unsupported pipeline: {args.pipeline}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()