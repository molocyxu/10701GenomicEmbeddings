# TF Embeddings Integration Summary

## What Was Done

### 1. **TF Embeddings Extraction** (`src/extract_tf_embeddings.py`)
- Loaded pre-computed NT-v2-50m embeddings:
  - **Gene embeddings**: 4912 genes × 512 dimensions
  - **TF embeddings**: 1833 TFs × 512 dimensions
- Mapped TF embeddings to perturbation IDs based on `tf_sequences.csv`:
  - Matched perturbation_id to TF gene_name or gene_id
  - Successfully mapped **1833 / 1837 perturbations** (99.8%)
  - 4 unmapped perturbations: ARNTL2, ARNTL, T, off-target (likely controls)
- **Output**: `artifacts/tf_embeddings.parquet` (1837 × 512)

### 2. **Updated Data Extraction** (`src/data_extraction.py`)
- Added functions to load and map TF embeddings during preprocessing
- TF embeddings are now saved as standard artifact alongside targets and covariates
- Can be re-run with: `python src/data_extraction.py`

### 3. **Updated All Three Baselines**
Each baseline now loads TF embeddings in addition to targets:
- **`src/baselines/no_change.py`**
- **`src/baselines/global_mean_shift.py`**
- **`src/baselines/additive_control.py`**

Changes:
- `load_artifacts()` now returns 3 items: `(gene_index, Y, tf_embeddings)`
- Each baseline logs: `"Loaded TF embeddings shape: (1837, 512)"`
- Predictions remain unchanged (baselines don't yet use embeddings)

## New Data Available

### Artifacts
```
artifacts/
├── gene_index.txt              (4912 genes)
├── targets.parquet             (1837 perturbs × 4912 genes)
├── covariates.parquet          (per-gene metadata)
└── tf_embeddings.parquet       (1837 perturbs × 512 dims) ✨ NEW
```

### TF Embeddings Structure
- **Index**: Perturbation IDs (matching `targets.index`)
- **Columns**: `emb_0, emb_1, ..., emb_511` (512 embedding dimensions)
- **Data**: Pre-computed NT-transformer embeddings from TF promoter sequences
- **Alignment**: 1833 valid embeddings; 4 NaN rows (unmapped perturbations)

## Baseline Results (Unchanged So Far)

All three baselines regenerated with TF embeddings loaded:

| Baseline | LOTO Pearson | 95% CI | Raw MSE |
|----------|------------|---------|---------|
| No-change | 0.0 | [0.0, 0.0] | 0.01194 |
| Global mean-shift | 0.0 | [0.0, 0.0] | 0.01194 |
| Additive-control | -0.2925 | [-0.2967, -0.2888] | 0.01195 |

**Note**: Predictions are identical to before because baselines don't yet use embeddings. The embeddings are now **available** for:
- Improved baseline implementations (TF similarity-based fallbacks)
- New model inputs (embedding features)
- Evaluation metrics (embedding-based consistency)

## Next Steps (Recommendations)

You can now enhance baselines using TF embeddings:

1. **Embedding-based similarity**: Replace tf_family lookup with cosine similarity
   - Use top-K nearest TFs by embedding similarity
   - Weighted average based on similarity scores

2. **Per-gene intercept baseline**: Fit per-gene offsets
   - Should improve centered Pearson metrics

3. **TF embedding + gene embedding model**: Supervised baseline
   - Use embeddings as direct model inputs

4. **Evaluation metric**: Embedding consistency
   - Check if similar TFs (by embedding) produce similar predictions

## Files Modified

- `src/extract_tf_embeddings.py` ✨ NEW
- `src/data_extraction.py` (updated)
- `src/baselines/no_change.py` (updated)
- `src/baselines/global_mean_shift.py` (updated)
- `src/baselines/additive_control.py` (updated)
