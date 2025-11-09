# ✅ Script Đã Sửa - run_evaluate.sh

## 🔴 LỖI TRONG SCRIPT CŨ:

### 1. Collection Name Mismatch (NGHIÊM TRỌNG)

**Build:**
```bash
--chunks data/chunks/chunks_token_380_50.jsonl
```
→ Tạo collection: `chunks_token_380_50_qwen_qwen3_embedding_0_6b`

**Evaluate (SAI):**
```bash
--collection chunks_recursive_380_50_qwen_qwen3_embedding_0_6b
```
→ ❌ Tìm collection `chunks_recursive` nhưng bạn build `chunks_token`

**Kết quả:** Script sẽ FAIL với lỗi "Collection not found"

---

### 2. Thiếu `--chunks_file` cho Hybrid Mode

**Hybrid mode YÊU CẦU:**
```bash
--retrieval_mode hybrid
--chunks_file data/chunks/chunks_token_380_50.jsonl  # ← THIẾU CÁI NÀY
--alpha 0.7
--rrf_k 60
```

Không có `--chunks_file` → không build được BM25 index → hybrid mode FAIL

---

## ✅ ĐÃ SỬA:

### Collection Names (Đúng)

| Chunks File | Embedding Model | Collection Name |
|------------|-----------------|-----------------|
| `chunks_token_380_50.jsonl` | `Qwen/Qwen3-Embedding-0.6B` | `chunks_token_380_50_qwen_qwen3_embedding_0_6b` ✅ |
| `chunks_token_380_50.jsonl` | `BAAI/bge-m3` | `chunks_token_380_50_baai_bge_m3` ✅ |

### Hybrid Mode Parameters (Đã thêm)

```bash
# Qwen - Hybrid
--retrieval_mode hybrid
--chunks_file data/chunks/chunks_token_380_50.jsonl  # ✅ Đã thêm
--alpha 0.7                                           # ✅ Đã thêm
--rrf_k 60                                            # ✅ Đã thêm

# BAAI - Hybrid  
--retrieval_mode hybrid
--chunks_file data/chunks/chunks_token_380_50.jsonl  # ✅ Đã thêm
--alpha 0.7                                           # ✅ Đã thêm
--rrf_k 60                                            # ✅ Đã thêm
```

---

## 📋 Script Flow (Đã Fix)

```
1. Build VectorStore
   ├─ Qwen/Qwen3-Embedding-0.6B
   │  → chunks_token_380_50_qwen_qwen3_embedding_0_6b
   │
   └─ BAAI/bge-m3
      → chunks_token_380_50_baai_bge_m3

2. Evaluate Qwen
   ├─ Dense mode   ✅
   └─ Hybrid mode  ✅ (với --chunks_file)

3. Evaluate BAAI
   ├─ Dense mode   ✅
   └─ Hybrid mode  ✅ (với --chunks_file)

4. Done! Results in CSV
```

---

## ⚙️ Parameters Summary

| Parameter | Value | Note |
|-----------|-------|------|
| `chunks_file` | `chunks_token_380_50.jsonl` | Token-based chunks |
| `top_k` | 5 | Top documents to retrieve |
| `retrieve_k` | 20 | For document-level eval |
| `agg_mode` | max | Aggregation: max score |
| `cosine_threshold` | 0.70 | Semantic matching |
| `eval_method` | both | Document + Semantic |
| `alpha` | 0.7 | Hybrid: 70% dense, 30% sparse |
| `rrf_k` | 60 | RRF fusion parameter |

---

## 🚀 Chạy Script

```bash
bash run_evaluate.sh
```

Hoặc nếu dùng Git Bash trên Windows:
```bash
bash run_evaluate.sh
```

---

## 📊 Expected Results

**2 models × 2 modes = 4 evaluations:**
- ✅ Qwen - Dense
- ✅ Qwen - Hybrid
- ✅ BAAI/bge-m3 - Dense
- ✅ BAAI/bge-m3 - Hybrid

**Output:** `data/evaluate_results/evaluation_results.csv`

---

## 💡 Nếu Muốn Dùng Python Script Thay Thế

Script `run_evaluate.py` sẽ an toàn hơn vì:
- ✅ Auto collection names (không typo)
- ✅ Build → Evaluate per model (lower memory)
- ✅ Incremental saves (safe if crash)

Edit `run_evaluate.py`:
```python
CHUNKS_FILE = "data/chunks/chunks_token_380_50.jsonl"  # ← Change this

EMBEDDING_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    # "google/embeddinggemma-300m",  # Comment out
]

EVAL_CONFIG = {
    "top_k": 5,
    "max_queries": 50,
    "retrieve_k": 20,
    "agg_mode": "max",
    "cosine_threshold": 0.70,
    "eval_method": "both",
    "alpha": 0.7,
    "rrf_k": 60,
}
```

Rồi chạy:
```bash
uv run run_evaluate.py
```
