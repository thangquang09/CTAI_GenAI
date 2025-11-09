# Evaluation Results Tracking Workflow

## 📊 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RUN EVALUATION                              │
│  uv run src/evaluate.py --mode from_collection --collection ... │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Load Collection       │
         │  Build Retriever       │
         │  Run Evaluation        │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │  Compute Metrics       │
         │  - hit_rate@K          │
         │  - recall@K            │
         │  - precision@K         │
         │  - mrr@K               │
         │  - ndcg@K              │
         │  - coverage@K          │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │  Extract Config        │
         │  - chunk_size          │
         │  - embedding_model     │
         │  - retrieval_mode      │
         │  - top_k, alpha, etc.  │
         └────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  save_evaluation_results()      │
    │                                 │
    │  1. Generate timestamp          │
    │     exp_20251109_170142         │
    │                                 │
    │  2. Save JSON config            │
    │     ├─ experiment_id            │
    │     ├─ timestamp                │
    │     ├─ pipeline config          │
    │     ├─ retriever config         │
    │     └─ evaluation config        │
    │                                 │
    │  3. Append to CSV               │
    │     ├─ timestamp, exp_id        │
    │     ├─ config (sorted)          │
    │     └─ metrics (sorted)         │
    └─────────┬───────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │  FILES CREATED                  │
    │                                 │
    │  📄 evaluation_results.csv      │
    │     (appended new row)          │
    │                                 │
    │  📄 exp_<timestamp>_config.json │
    │     (new file)                  │
    └─────────┬───────────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │  ANALYSIS OPTIONS               │
    └─────────┬───────────────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
  ┌─────────┐  ┌──────────────┐
  │ Script  │  │ Manual (py)  │
  └────┬────┘  └──────┬───────┘
       │              │
       ▼              ▼
  analyze_results.py   pandas
       │              │
       ├── Latest     ├── Custom queries
       ├── Compare    ├── Visualizations
       ├── Top 5      ├── Statistics
       └── Summary    └── Export
```

---

## 🔄 Workflow Steps

### 1️⃣ Run Evaluation
```bash
uv run src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --top_k 5 \
  --eval_method both
```

### 2️⃣ Auto-save (Happens Automatically)
```
FINAL EVALUATION SUMMARY
================================================================================
Queries evaluated: 50
Evaluation method: both
Top-K: 5

Metrics:
  hit_rate@5                     0.6400
  recall@5                       0.4500
  ...

[SAVED] Config: data/evaluate_results/exp_20251109_170142_config.json
[SAVED] Results: data/evaluate_results/evaluation_results.csv

================================================================================
RESULTS SAVED SUCCESSFULLY
================================================================================
  CSV: data/evaluate_results/evaluation_results.csv
  JSON: data/evaluate_results/exp_20251109_170142_config.json
================================================================================
```

### 3️⃣ Analyze Results
**Option A: Use script**
```bash
uv run data/evaluate_results/analyze_results.py
```

**Option B: Manual Python**
```python
import pandas as pd

df = pd.read_csv('data/evaluate_results/evaluation_results.csv')

# Latest experiment
print(df.tail(1))

# Compare chunk sizes
df.groupby('chunk_size')[['hit_rate@5', 'mrr@5']].mean()

# Best configuration
best = df.loc[df['hit_rate@5'].idxmax()]
print(f"Best config: {best['collection_name']}")
print(f"Hit rate: {best['hit_rate@5']:.4f}")
```

---

## 📁 File Tracking

### CSV Structure
```csv
timestamp,experiment_id,chunk_size,embedding_model,...,hit_rate@5,mrr@5,...
20251109_143000,exp_20251109_143000,380,BAAI/bge-small-en-v1.5,...,0.64,0.4567,...
20251109_150000,exp_20251109_150000,500,BAAI/bge-small-en-v1.5,...,0.68,0.5012,...
20251109_153000,exp_20251109_153000,380,sentence-transformers/all-MiniLM-L6-v2,...,0.62,0.4321,...
```

### JSON Config (per experiment)
```json
{
  "experiment_id": "exp_20251109_170142",
  "timestamp": "20251109_170142",
  "collection_name": "chunks_recursive_380_50_baai_bge_small_en_v1_5",
  "chunk_size": 380,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "retrieval_mode": "dense",
  "top_k": 5,
  "eval_method": "both",
  ...
}
```

---

## 🎯 Experiment Workflow Example

### Scenario: Find best chunk size

**Step 1: Baseline (chunk_size=380)**
```bash
uv run src/evaluate.py --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --top_k 5
# → exp_20251109_140000
```

**Step 2: Try chunk_size=500**
```bash
uv run src/evaluate.py --mode full \
  --strategy recursive --chunk_size 500 --overlap 50 --top_k 5
# → exp_20251109_141000
```

**Step 3: Try chunk_size=300**
```bash
uv run src/evaluate.py --mode full \
  --strategy recursive --chunk_size 300 --overlap 50 --top_k 5
# → exp_20251109_142000
```

**Step 4: Compare**
```python
import pandas as pd
df = pd.read_csv('data/evaluate_results/evaluation_results.csv')

# Filter by strategy
recursive_results = df[df['strategy'] == 'recursive']

# Compare chunk sizes
comparison = recursive_results.groupby('chunk_size').agg({
    'hit_rate@5': 'mean',
    'mrr@5': 'mean',
    'recall@5': 'mean',
    'precision@5': 'mean',
}).round(4)

print(comparison)
```

**Output**:
```
             hit_rate@5   mrr@5  recall@5  precision@5
chunk_size                                            
300              0.6200  0.4321    0.4300       0.3100
380              0.6400  0.4567    0.4500       0.3200
500              0.6800  0.5012    0.4900       0.3500
```

**Conclusion**: chunk_size=500 performs best!

---

## 🔬 Advanced Analysis Examples

### 1. Compare Embedding Models
```python
df.groupby('embedding_model')[['hit_rate@5', 'mrr@5']].mean()
```

### 2. Compare Retrieval Modes (Dense vs Hybrid)
```python
df.groupby('retrieval_mode')[['hit_rate@5', 'mrr@5']].mean()
```

### 3. Find Optimal Alpha for Hybrid
```python
hybrid = df[df['retrieval_mode'] == 'hybrid']
hybrid.plot(x='alpha', y=['hit_rate@5', 'mrr@5'], marker='o')
```

### 4. Experiment Timeline
```python
df.plot(x='timestamp', y=['hit_rate@5', 'mrr@5'], marker='o')
plt.xticks(rotation=45)
plt.title('Metrics Over Time')
```

### 5. Best Overall Configuration
```python
# Multi-criteria ranking
df['score'] = df['hit_rate@5'] * 0.4 + df['mrr@5'] * 0.6
best = df.loc[df['score'].idxmax()]

print(f"Best Configuration:")
print(f"  Experiment: {best['experiment_id']}")
print(f"  Collection: {best['collection_name']}")
print(f"  Chunk size: {best['chunk_size']}")
print(f"  Embedding: {best['embedding_model']}")
print(f"  Mode: {best['retrieval_mode']}")
print(f"  Hit rate: {best['hit_rate@5']:.4f}")
print(f"  MRR: {best['mrr@5']:.4f}")
```

---

## 🎓 Tips & Best Practices

### ✅ DO:
- Run baseline first (default config)
- Change ONE variable at a time
- Use meaningful collection names
- Document findings in notes
- Regular analysis to track progress

### ❌ DON'T:
- Change multiple variables at once
- Delete CSV without backup
- Ignore failed experiments
- Forget to analyze results
- Run without proper config

### 📊 Recommended Metrics Priority:
1. **hit_rate@5** - Most important (did we find ANY relevant article?)
2. **mrr@5** - Quality of ranking
3. **recall@5** - Coverage of relevant articles
4. **precision@5** - Relevance ratio

---

## 🚨 Troubleshooting

**Q: CSV not created?**
→ Run at least one evaluation first

**Q: Config extraction failed?**
→ Ensure collection name follows pattern: `chunks_{strategy}_{size}_{overlap}_{model}`

**Q: Want to reset experiments?**
```bash
# BACKUP FIRST!
cp data/evaluate_results/evaluation_results.csv backup.csv

# Clear (or delete specific rows in CSV)
rm data/evaluate_results/evaluation_results.csv
rm data/evaluate_results/exp_*.json
```

**Q: Need custom analysis?**
→ Use pandas directly with the CSV file

---

## 📈 Next Steps

1. **Visualizations**: Add matplotlib/plotly plots
2. **Statistical tests**: Add significance testing
3. **Dashboard**: Build Streamlit/Gradio UI
4. **Reports**: Auto-generate LaTeX/PDF reports
5. **Monitoring**: Track experiments over time

---

## 🎉 Summary

This workflow provides:
- ✅ **Automatic tracking** - No manual effort
- ✅ **Full reproducibility** - All config saved
- ✅ **Easy analysis** - CSV + scripts
- ✅ **Systematic experiments** - Compare configs
- ✅ **Data integrity** - Append-only, no overwrites
