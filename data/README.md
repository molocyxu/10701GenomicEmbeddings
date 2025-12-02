## Data

### 1. Reference Genome and Annotation

- **`data/Homo_sapiens.GRCh38.110.gtf`**  
  - **Type**: GTF gene annotation (Ensembl GRCh38, release 110).  
  - **Content**: Genomic coordinates, exon structures, transcript IDs, gene biotypes, and strand information for human genes and transcripts.  
- **`data/Homo_sapiens.GRCh38.dna.primary_assembly.fa`**   
  - **Type**: FASTA file with the primary assembly of the human reference genome (GRCh38).  
  - **Content**: Chromosome‑level DNA sequences (A/C/G/T/N).  

---

### 2. Perturb‑seq CRISPRa Expression Data

- **`data/Hs27_fibroblast_CRISPRa_mean_pop.h5ad`**  
  - **Type**: `AnnData` object (`.h5ad`).  
  - **Content** (typical structure, exact fields may vary):  
    - `X`: Gene expression matrix, likely containing **mean expression profiles per perturbation / cell population** in the Hs27 CRISPRa screen.  
    - `obs`: Metadata for each observation (e.g. targeted gene, gRNA identifiers, experimental batch, cell type).  
    - `var`: Gene‑level metadata (e.g. Ensembl ID, gene symbol, feature type).  

---

### 3. Gene Sequence Tables

These CSVs provide promoter‑centric genomic sequences around gene TSSs, which are used as input to the nucleotide transformer model.

- **`Hs27_gene_sequences_12kb_4912genes.csv`**  
  - **Type**: CSV table for Hs27‑perturbed genes.  
  - **Typical columns**:  
    - `gene_id`: Ensembl gene ID (e.g. `ENSG00000000419`).  
    - `chrom`: Chromosome of the gene.  
    - `tss`: Genomic coordinate of the transcription start site.  
    - `strand`: `+` or `-` DNA strand.  
    - `sequence`: DNA sequence string (A/C/G/T/N) representing a **12 kb window around the TSS**.  
  - **Content**: 4,912 Hs27 genes targeted in the CRISPRa screen, each with a 12 kb TSS‑centered sequence.  

---

### 4. Transcription Factor (TF) Promoter Sequences

- **`tf_sequences.csv`**  
  - Note that we have **dropped 3 TFs** that are not mapped.
  - **Type**: CSV with promoter sequences for selected transcription factor (TF) genes.  
  - **Typical columns** (based on file header):  
    - `gene_name`: TF gene symbol (e.g. `PRDM16`).  
    - `gene_id`: Ensembl gene ID of the TF.  
    - `chrom`: Chromosome.  
    - `tss`: TSS coordinate of the TF gene.  
    - `strand`: `+` or `-` strand.  
    - `sequence`: DNA sequence string around the TF TSS (fixed‑length window similar to gene promoters).  
  

---

### 5. Nucleotide‑Transformer Embeddings

NT-v2-50m produces four NumPy files:

- **`gene_embeddings.npy`**  
  - **Shape**: `(N_genes, embedding_dim)`
  - **Content**:  
    - Mean‑pooled nucleotide‑transformer embeddings for each gene’s 12 kb promoter sequence.  
- **`gene_embeddings_gene_ids.npy`**  
  - **Shape**: `(N_genes,)`.  
  - **Content**:  
    - Ensembl gene IDs in the **exact same order** as rows in `gene_embeddings.npy`.  
- **`tf_embeddings.npy`**  
  - **Shape**: `(N_tfs, embedding_dim)`
  - **Content**:  
    - Mean‑pooled nucleotide‑transformer embeddings for each TF promoter sequence.  
- **`tf_embeddings_gene_ids.npy`**  
  - **Shape**: `(N_tfs,)`.  
  - **Content**:  
    - Ensembl gene IDs (or other TF identifiers) corresponding to rows in `tf_embeddings.npy`, aligned with `tf_sequences.csv`.  

