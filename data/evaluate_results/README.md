# Evaluation Results

This directory contains evaluation results from the RAG retrieval system.

## File Structure

### 1. `evaluation_results.csv`
Main CSV file containing all evaluation metrics across experiments. Each row represents one experiment run.

**Columns**:
- `timestamp`: Timestamp of experiment (format: YYYYMMDD_HHMMSS)
- `experiment_id`: Unique experiment ID (format: exp_YYYYMMDD_HHMMSS)

**Pipeline Configuration**:
- `mode`: Pipeline mode (`full` or `from_collection`)
- `collection_name`: Qdrant collection name
- `strategy`: Chunking strategy (`recursive`, `token`, etc.)
- `chunk_size`: Chunk size in characters/tokens
- `overlap`: Overlap size
- `embedding_model`: HuggingFace embedding model used
- `qdrant_path`: Path to Qdrant database
- `retrieval_mode`: Retrieval mode (`dense` or `hybrid`)

**Retriever Configuration**:
- `top_k`: Number of documents to retrieve
- `search_type`: Search type (`similarity`, `mmr`, `similarity_score_threshold`)
- `alpha`: Hybrid mode weight (0=sparse, 1=dense)
- `rrf_k`: RRF parameter for hybrid mode

**Evaluation Configuration**:
- `eval_method`: Evaluation method (`document`, `semantic`, or `both`)
- `num_queries`: Number of queries evaluated
- `max_queries`: Max queries limit (0 = all)
- `retrieve_k`: Number of chunks retrieved for document-level evaluation
- `agg_mode`: Aggregation mode (`max`, `sum`, `mean`)
- `cosine_threshold`: Cosine similarity threshold for semantic evaluation

**Metrics (Document-Level)**:
- `hit_rate@K`: Tỷ lệ queries có ít nhất 1 bài đúng trong top-K
- `recall@K`: Tỷ lệ bài đúng được tìm thấy / tổng số bài đúng
- `precision@K`: Tỷ lệ bài đúng trong top-K / K
- `mrr@K`: Mean Reciprocal Rank
- `ndcg@K`: Normalized Discounted Cumulative Gain
- `coverage@K`: Coverage metric

**Metrics (Chunk-Level Semantic)**:
- `hit_rate@K_semantic`: Hit rate for semantic evaluation
- `recall@K_semantic`: Recall for semantic evaluation
- `precision@K_semantic`: Precision for semantic evaluation
- `mrr@K_semantic`: MRR for semantic evaluation
- `ndcg@K_semantic`: nDCG for semantic evaluation
- `coverage@K_semantic`: Coverage for semantic evaluation

### 2. `exp_<timestamp>_config.json`
Individual JSON config files for each experiment.

**Structure**:
```json
{
  "experiment_id": "exp_20251109_143000",
  "timestamp": "20251109_143000",
  "mode": "from_collection",
  "collection_name": "chunks_recursive_380_50_baai_bge_small_en_v1_5",
  "strategy": "recursive",
  "chunk_size": 380,
  "overlap": 50,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "qdrant_path": "langchain_qdrant",
  "retrieval_mode": "dense",
  "top_k": 5,
  "search_type": "similarity",
  "alpha": 0.5,
  "rrf_k": 60,
  "eval_method": "both",
  "num_queries": 50,
  "max_queries": 0,
  "retrieve_k": 50,
  "agg_mode": "max",
  "cosine_threshold": 0.75
}
```

## Usage

### Run Evaluation
```bash
# Full pipeline
uv run src/evaluate.py \
  --mode full \
  --strategy recursive \
  --chunk_size 380 \
  --overlap 50 \
  --top_k 5

# From existing collection
uv run src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --top_k 5
```

### Analyze Results
```python
import pandas as pd

# Load all results
df = pd.read_csv('data/evaluate_results/evaluation_results.csv')

# Compare different chunk sizes
chunk_comparison = df.groupby('chunk_size')[['hit_rate@5', 'mrr@5']].mean()

# Compare embedding models
model_comparison = df.groupby('embedding_model')[['hit_rate@5', 'mrr@5']].mean()

# Compare retrieval modes
mode_comparison = df.groupby('retrieval_mode')[['hit_rate@5', 'mrr@5']].mean()
```

## Notes

- CSV is append-only - new experiments are added as new rows
- Each experiment gets a unique JSON config file for full reproducibility
- Timestamp format: YYYYMMDD_HHMMSS
- All metrics are in range [0, 1]
- Higher values = better performance
