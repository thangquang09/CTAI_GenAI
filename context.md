# Project Context: CTAI_GenAI - WixQA Chunking Pipeline

## 📌 Overview
Dự án xây dựng pipeline để chunk WixQA corpus thành các đoạn văn bản nhỏ hơn phục vụ cho RAG (Retrieval-Augmented Generation) hoặc các tác vụ NLP khác.

---

## 🗂️ Cấu trúc Project

### 1. **src/chunking/** - Module xử lý chunking
Chứa logic chính để chia nhỏ văn bản từ WixQA dataset.

#### **chunk_wixqa.py** - Script chính
- **Chức năng**: Load WixQA corpus và chia thành chunks theo nhiều chiến lược
- **Dataset**: `Wix/WixQA` (split: `wix_kb_corpus`)
- **Input fields**: `id`, `title`, `url`, `article_type`, `contents`
- **Output format**: JSONL với cấu trúc:
  ```json
  {
    "chunk_id": "article_id::position",
    "article_id": "original_article_id",
    "title": "...",
    "url": "...",
    "article_type": "...",
    "text": "chunk content",
    "position": 0
  }
  ```

**Arguments**:
- `--strategy`: Chiến lược chunking (`recursive`, `token`, `semantic`, `hybrid_para_token`)
- `--chunk_size`: Kích thước chunk (default: 380)
- `--overlap`: Số token overlap (default: 50)
- `--max_records`: Giới hạn số bài xử lý (0 = toàn bộ)
- `--out`: Đường dẫn output file (mặc định: `data/chunks/chunks_{strategy}_{chunk_size}_{overlap}.jsonl`)

**Usage**:
```bash
python src/chunking/chunk_wixqa.py --strategy recursive --chunk_size 380 --overlap 50
python src/chunking/chunk_wixqa.py --strategy token --chunk_size 380 --overlap 50
```

#### **split_function.py** - Các hàm split
- **`split_recursive()`**: Sử dụng `RecursiveCharacterTextSplitter` từ LangChain
  - Separators: `["\n\n", "\n", ". ", " ", ""]`
  - Return: `List[str]`
  
- **`split_token()`**: Sử dụng `TokenTextSplitter` từ LangChain
  - Chia theo số lượng token
  - Return: `List[str]`
  
- **`split_semantic()`**: Chưa implement (placeholder)

---

## 📊 Data Output

### **data/chunks/** - Thư mục chứa kết quả chunking

#### **chunks_recursive_380_50.jsonl**
- Strategy: Recursive Character Text Splitting
- Chunk size: 380 characters
- Overlap: 50 characters
- **Số chunks**: 59,049 chunks
- Đặc điểm: Chia theo các separators tự nhiên (paragraph, sentence, space)

#### **chunks_token_380_50.jsonl**
- Strategy: Token-based splitting
- Chunk size: 380 tokens
- Overlap: 50 tokens
- **Số chunks**: 12,608 chunks
- Đặc điểm: Chia theo số lượng token, phù hợp cho embedding models

**Sample chunk**:
```json
{
  "chunk_id": "860475a2cbc65c226ecf08729d0430584e46f35f23a3e10bb5500c2c4ad09168::0",
  "article_id": "860475a2cbc65c226ecf08729d0430584e46f35f23a3e10bb5500c2c4ad09168",
  "title": "Wix Events: About the Event Details and Registration Form Pages",
  "url": "https://support.wix.com/en/article/wix-events-about-the-event-details-and-registration-form-pages",
  "article_type": "article",
  "text": "Wix Events: About the Event Details and Registration Form Pages...",
  "position": 0
}
```

---

## 🔧 Tech Stack
- **Language**: Python 3.10+
- **Libraries**: 
  - `datasets` (HuggingFace) - Load WixQA dataset
  - `langchain-text-splitters` - Text chunking strategies
  - `langchain-qdrant` - Qdrant vector store integration
  - `langchain-huggingface` - HuggingFace embeddings
  - `qdrant-client` - Qdrant Python client
  - `tqdm` - Progress bars
  - Standard: `argparse`, `json`, `os`, `re`, `typing`, `uuid`

---

## ⚠️ Notes & TODOs

### Fixed Issues:
1. ✅ **Deprecated imports**: Đã chuyển từ `langchain-community` sang `langchain-qdrant`
2. ✅ **UUID requirement**: Qdrant yêu cầu UUID IDs → dùng UUID5 generation
3. ✅ **Client lock conflicts**: Fix bằng cách dùng single client instance
4. ✅ **Collection not found**: Fix bằng `from_documents()` API pattern

### Current Issues:
1. **Type hints in chunking**: `Dict[str, any]` cần sửa thành `Dict[str, Any]` (viết hoa)
2. **Incomplete strategies**: `semantic`, `hybrid_para_token` chưa implement (chỉ có `pass`)
3. **Error handling**: Thiếu try-except cho file I/O và dataset loading
4. **Dead code**: Comment `# "n_tokens": count_tokens(p),` nên xóa hoặc implement

### Recommendations:
- Thêm logging để track progress
- Implement các strategy còn lại
- Thêm validation cho output
- Consider batch processing cho dataset lớn
- Thêm metrics: avg chunk length, token distribution
- Add retry logic cho embedding API calls
- Implement collection backup/restore

---

---

## �️ Vector Store Setup

### **src/vectorstore/** - Module xây dựng vector database

#### **build_vectordb.py** - Script tạo Qdrant vector store
- **Chức năng**: Index chunks vào Qdrant local vector database
- **Input**: JSONL chunks từ `data/chunks/`
- **Output**: Qdrant collection trong `langchain_qdrant/`
- **Vector Store**: Qdrant (local on-disk storage)
- **Embeddings**: HuggingFace models (default: `BAAI/bge-small-en-v1.5`)

**Arguments**:
- `--chunks`: Đường dẫn file JSONL chunks (required)
- `--out_dir`: Thư mục lưu Qdrant database (default: `langchain_qdrant`)
- `--collection`: Tên collection (default: auto từ tên file chunks + embedding model)
- `--embedding_model`: HuggingFace embedding model (default: `BAAI/bge-small-en-v1.5`)
- `--limit`: Giới hạn số chunks để index (0 = toàn bộ)
- `--recreate`: Flag để xóa và tạo mới collection
- `--batch_size`: Kích thước batch khi index (default: 256)
- `--grpc`: Dùng gRPC cho local client (thường không cần)

**Usage**:
```bash
# Tự động collection name: chunks_recursive_380_50_baai_bge_small_en_v1_5
uv run src/vectorstore/build_vectordb.py --chunks data/chunks/chunks_recursive_380_50.jsonl

# Với recreate và custom embedding model
uv run src/vectorstore/build_vectordb.py --chunks data/chunks/chunks_token_380_50.jsonl --embedding_model "BAAI/bge-base-en-v1.5" --recreate
# → Collection: chunks_token_380_50_baai_bge_base_en_v1_5

# Với sentence-transformers model
uv run src/vectorstore/build_vectordb.py --chunks data/chunks/chunks_recursive_380_50.jsonl --embedding_model "sentence-transformers/all-MiniLM-L6-v2"
# → Collection: chunks_recursive_380_50_sentence_transformers_all_minilm_l6_v2

# Custom collection name (vẫn append embedding model)
uv run src/vectorstore/build_vectordb.py --chunks data/chunks/chunks_recursive_380_50.jsonl --collection my_custom --limit 1000
# → Collection: my_custom_baai_bge_small_en_v1_5
```

**Key Features**:
- ✅ Sử dụng `langchain-qdrant` package mới (không dùng deprecated `langchain-community`)
- ✅ Tự động generate UUID cho point IDs từ `chunk_id` (sử dụng UUID5 để deterministic)
- ✅ Batch processing với progress bar
- ✅ **Auto collection name từ tên file chunks + embedding model name**
- ✅ **Sanitize collection name** để tránh ký tự đặc biệt (`/`, `-`, `.` → `_`)
- ✅ Metadata đầy đủ: `chunk_id`, `article_id`, `title`, `url`, `article_type`, `position`
- ✅ Tránh client lock conflicts bằng cách dùng single client instance

**Technical Details**:
- **Collection Naming**: `{chunks_name}_{embedding_model_sanitized}`
  - Ví dụ: `chunks_recursive_380_50_baai_bge_small_en_v1_5`
  - Sanitize: `BAAI/bge-small-en-v1.5` → `baai_bge_small_en_v1_5`
  - Lowercase, loại bỏ ký tự đặc biệt, giới hạn 255 chars
- **UUID Generation**: Dùng UUID5 với fixed namespace để convert `chunk_id` → UUID
  - Cùng `chunk_id` luôn tạo ra cùng UUID (idempotent)
  - Original `chunk_id` vẫn được lưu trong metadata
- **Vector Store API**: 
  - Batch đầu: `QdrantVectorStore.from_documents()` → tạo collection
  - Batch tiếp: `vector_store.add_documents()` → thêm vào collection
- **Collection Info**: Hiển thị status, vector count (không có vector size nữa do API change)

**Dependencies**:
```bash
uv pip install langchain-qdrant langchain-huggingface
```

---

## 🔍 Retriever Setup

### **src/vectorstore/** - Module retrieval

#### **build_retriever.py** - Script tạo retriever từ Qdrant collection
- **Chức năng**: Load Qdrant collection và tạo LangChain retriever
- **Input**: Qdrant collection có sẵn
- **Output**: BaseRetriever object sẵn sàng cho RAG
- **Search modes**: similarity, MMR, similarity_score_threshold

**Arguments**:
- `--collection`: Tên collection trong Qdrant (required)
- `--qdrant_path`: Đường dẫn Qdrant database (default: `langchain_qdrant`)
- `--embedding_model`: HuggingFace embedding model - **phải khớp với model khi build** (default: `BAAI/bge-small-en-v1.5`)
- `--query`: Test query để thử search (optional)
- `--top_k`: Số documents trả về (default: 5)
- `--search_type`: Loại search - `similarity`, `mmr`, `similarity_score_threshold` (default: `similarity`)
- `--score_threshold`: Score threshold cho similarity_score_threshold mode
- `--grpc`: Dùng gRPC cho local client

**Usage**:
```bash
# Build retriever và test search
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --query "How to create an event in Wix?" \
  --top_k 3

# Với MMR search (đa dạng hóa kết quả)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --search_type mmr \
  --top_k 10

# Với score threshold
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --search_type similarity_score_threshold \
  --score_threshold 0.7
```

**Key Features**:
- ✅ Load existing Qdrant collections với validation
- ✅ 3 search modes: similarity, MMR, similarity_score_threshold
- ✅ Configurable search parameters (top_k, thresholds, lambda_mult cho MMR)
- ✅ Collection existence check với helpful error messages
- ✅ Tương thích hoàn toàn với LangChain ecosystem (`.invoke()` API)
- ✅ CLI để test retrieval nhanh
- ✅ Helper function `search_documents()` để dễ sử dụng trong code

**API Usage trong code**:
```python
from src.vectorstore.build_retriever import build_retriever

# Basic similarity search
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    search_kwargs={"k": 5}
)
docs = retriever.invoke("How to create events?")

# MMR search (đa dạng hóa)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.5}
)
```

**Search Types**:
- **`similarity`**: Vector cosine similarity (default) - tìm docs gần nhất
- **`mmr`**: Maximal Marginal Relevance - cân bằng giữa relevance và diversity
- **`similarity_score_threshold`**: Chỉ trả về docs có score >= threshold

---

## 🧪 Evaluation Pipeline

### **src/** - Evaluation module

#### **evaluate.py** - Script đánh giá RAG retrieval system
- **Chức năng**: End-to-end pipeline từ chunking → vectorstore → retriever → evaluation
- **Input**: WixQA dataset hoặc Qdrant collection có sẵn
- **Output**: Retriever sẵn sàng + evaluation metrics
- **Pipeline modes**: 2 modes linh hoạt

**Pipeline Modes**:

1. **`full`**: Pipeline hoàn chỉnh
   - Step 1: Chunking → tạo JSONL chunks
   - Step 2: Build vector store → tạo Qdrant collection
   - Step 3: Build retriever → load retriever
   - Step 4: Evaluate với metrics

2. **`from_collection`**: Load từ collection có sẵn
   - Step 1: Build retriever từ collection
   - Step 2: Evaluate với metrics

**Evaluation Methods**:

### **1. Document-Level Evaluation (Hướng 1)**

Gom chunks theo `article_id` rồi tính metrics ở mức document.

**Ý tưởng**:
- Retriever trả về nhiều chunks (top-50 chẳng hạn)
- Gom chunks theo `article_id` với score aggregation (max/sum/mean)
- Rank các articles theo aggregated score
- Lấy top-K articles và tính metrics so với ground truth

**Metrics @K**:
- **Hit Rate @K**: Tỷ lệ queries có ít nhất 1 bài đúng trong top-K
- **Recall @K**: Tỷ lệ bài đúng được tìm thấy / tổng số bài đúng
- **Precision @K**: Tỷ lệ bài đúng trong top-K / K
- **MRR @K**: Mean Reciprocal Rank - trung bình 1/rank của bài đúng đầu tiên

**Aggregation modes**:
- `max`: Score = max(chunk_scores) - chunk tốt nhất đại diện
- `sum`: Score = sum(chunk_scores) - tổng độ liên quan
- `mean`: Score = mean(chunk_scores) - độ liên quan trung bình

### **2. Chunk-Level Semantic Evaluation (Hướng 2)**

Đánh giá chunks dựa trên cosine similarity với gold embeddings.

**Ý tưởng**:
- Tạo gold embeddings cho mỗi article bằng mean-pooling các chunk embeddings
- So sánh embedding của mỗi retrieved chunk với gold embeddings
- Chunk được coi là "đúng" nếu `cosine_similarity(chunk_vec, gold_vec) >= threshold`
- Tính metrics dựa trên correctness của chunks

**Metrics @K**:
- **Hit Rate @K**: Tỷ lệ queries có ít nhất 1 chunk đúng trong top-K
- **Recall @K**: Tỷ lệ bài đúng được "cover" bởi chunks đúng / tổng số bài đúng
- **Precision @K**: Tỷ lệ chunks đúng / K
- **MRR @K**: Mean Reciprocal Rank của chunk đúng đầu tiên

**Hyperparameters**:
- `cosine_threshold`: Ngưỡng để coi chunk là đúng (default: 0.75)

**Arguments**:

*Mode selection*:
- `--mode`: Pipeline mode - `full` hoặc `from_collection` (default: `full`)

*Chunking params (cho mode=full)*:
- `--strategy`: Chiến lược chunking - `recursive`, `token` (default: `recursive`)
- `--chunk_size`: Kích thước chunk (default: 380)
- `--overlap`: Overlap size (default: 50)
- `--max_records`: Giới hạn số bài để chunk - 0=toàn bộ (default: 0)

*Vector store params*:
- `--qdrant_path`: Đường dẫn Qdrant database (default: `langchain_qdrant`)
- `--embedding_model`: HuggingFace embedding model (default: `BAAI/bge-small-en-v1.5`)
- `--collection`: Tên collection - **required cho mode=from_collection**
- `--recreate`: Flag xóa và tạo lại collection (chỉ dùng với mode=full)

*Retriever params*:
- `--top_k`: Số documents trả về (default: 5)
- `--search_type`: Loại search - `similarity`, `mmr`, `similarity_score_threshold` (default: `similarity`)

*Evaluation params*:
- `--eval_method`: Phương pháp đánh giá - `document`, `semantic`, `both` (default: `both`)
- `--max_queries`: Giới hạn số queries để đánh giá - 0=toàn bộ (default: 0)
- `--retrieve_k`: Số chunks lấy cho document-level eval (default: 50)
- `--agg_mode`: Aggregation mode - `max`, `sum`, `mean` (default: `max`)
- `--cosine_threshold`: Threshold cho semantic eval (default: 0.75)

**Usage**:

```bash
# 1. Full pipeline với cả 2 evaluation methods
uv run python src/evaluate.py \
  --mode full \
  --strategy recursive \
  --chunk_size 380 \
  --overlap 50 \
  --max_records 100 \
  --eval_method both \
  --max_queries 50 \
  --top_k 5

# 2. Evaluate từ collection có sẵn (chỉ document-level)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --eval_method document \
  --top_k 5 \
  --retrieve_k 50 \
  --agg_mode max

# 3. Evaluate semantic với custom threshold
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_token_380_50_baai_bge_small_en_v1_5 \
  --eval_method semantic \
  --top_k 10 \
  --cosine_threshold 0.8

# 4. Full evaluation với custom embedding model
uv run python src/evaluate.py \
  --mode full \
  --strategy token \
  --chunk_size 512 \
  --overlap 64 \
  --embedding_model "BAAI/bge-base-en-v1.5" \
  --eval_method both \
  --max_queries 100

# 5. Final
uv run src/evaluate.py --mode from_collection --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 --qdrant_path langchain_qdrant --embedding_model BAAI/bge-small-en-v1.5 --top_k 5 --max_queries 5 --eval_method both --retrieve_k 20 --agg_mode max --cosine_threshold 0.70

```
**Key Features**:
- ✅ **2 evaluation methods**: Document-level (chunk aggregation) và Chunk-level semantic (cosine similarity)
- ✅ **4 metrics**: Hit Rate, Recall, Precision, MRR (cho cả 2 methods)
- ✅ **Flexible evaluation**: Có thể chạy riêng hoặc cả 2 methods
- ✅ **Gold embeddings**: Tự động build từ chunks trong collection bằng mean-pooling
- ✅ **Re-embedding approach**: Embed lại retrieved chunks để đảm bảo consistency
- ✅ **Configurable params**: Threshold, aggregation mode, K values
- ✅ **Progress tracking**: tqdm progress bars cho mỗi step
- ✅ **Debug statistics**: Hiển thị similarity range và embedding stats
- ✅ **Professional output**: Clean formatting không có emoji quá nhiều
- ✅ **Integration**: Kết nối với full pipeline (chunking → vectorstore → retriever)

**Output Example**:
```
================================================================================
LOADING EXISTING COLLECTION
================================================================================

================================================================================
STEP 3: BUILD RETRIEVER
================================================================================
Loading collection: chunks_recursive_380_50_baai_bge_small_en_v1_5
   Status: green
   Vectors: 59049
[SUCCESS] Retriever ready!
   Search type: similarity
   Search kwargs: {'k': 5}

================================================================================
[SUCCESS] RETRIEVER READY
================================================================================


================================================================================
LOADING EVALUATION DATASET
================================================================================
Loaded 50 queries from split: wixqa_expertwritten
================================================================================

================================================================================
RUNNING EVALUATION
================================================================================

================================================================================
EVALUATING: DOCUMENT-LEVEL (CHUNK AGGREGATION)
================================================================================
Config: top_k=5, retrieve_k=50, agg_mode=max
Evaluating queries: 100%|████████| 50/50 [00:06<00:00,  8.10it/s]

--------------------------------------------------------------------------------
DOCUMENT-LEVEL RESULTS:
  hit_rate@5: 0.6400
  recall@5: 0.5867
  precision@5: 0.1360
  mrr@5: 0.4567
--------------------------------------------------------------------------------

================================================================================
EVALUATING: CHUNK-LEVEL SEMANTIC (COSINE SIMILARITY)
================================================================================
Config: top_k=5, cosine_threshold=0.7

Building gold embeddings for 64 articles...
Building gold embeddings: 100%|████████| 64/64 [01:20<00:00,  1.25s/it]
[SUCCESS] Built 64 gold embeddings
Evaluating queries: 100%|████████| 50/50 [00:14<00:00,  3.36it/s]

Debug Statistics:
  Total chunks evaluated: 250
  Chunks embedded: 250
  Similarity range: [0.6427, 1.0000]
  Threshold used: 0.7

--------------------------------------------------------------------------------
CHUNK-LEVEL SEMANTIC RESULTS:
  hit_rate@5_semantic: 1.0000
  recall@5_semantic: 0.5867
  precision@5_semantic: 0.9560
  mrr@5_semantic: 0.9567
--------------------------------------------------------------------------------

================================================================================
FINAL EVALUATION SUMMARY
================================================================================
Queries evaluated: 50
Evaluation method: both
Top-K: 5

Metrics:
  hit_rate@5                     0.6400
  recall@5                       0.5867
  precision@5                    0.1360
  mrr@5                          0.4567
  hit_rate@5_semantic            1.0000
  recall@5_semantic              0.5867
  precision@5_semantic           0.9560
  mrr@5_semantic                 0.9567
================================================================================
```

**Technical Details**:

1. **Document-Level Aggregation**:
   - Retrieve top-`retrieve_k` chunks (e.g., 50)
   - Group by `article_id` và aggregate scores
   - Rank articles theo aggregated score
   - Lấy top-K articles để tính metrics
   - Score giả định: inverse rank (1/position) nếu retriever không trả về scores

2. **Semantic Evaluation**:
   - Pre-compute gold embeddings: mean-pool all chunks per article
   - Retrieve top-K chunks
   - Re-embed retrieved chunk text using same embedding model
   - Compute max cosine similarity với gold embeddings
   - Label chunk = "correct" if `max_sim >= threshold`
   - Compute metrics dựa trên labels
   - **Note**: Re-embedding approach được sử dụng thay vì retrieve vectors từ Qdrant để đảm bảo consistency

3. **Dataset**:
   - WixQA `wixqa_expertwritten` split
   - Format: `{question, answer, article_ids}`
   - Ground truth: list of article IDs (có thể nhiều bài đúng/query)

---

## 📝 Development Log

### Completed:
- ✅ Setup chunking pipeline với 2 strategies (recursive, token)
- ✅ Generate chunks cho toàn bộ WixQA corpus (59K recursive, 12.6K token)
- ✅ Output format chuẩn hóa với metadata đầy đủ
- ✅ **Vector store setup với Qdrant local**
- ✅ **Auto UUID generation cho Qdrant point IDs**
- ✅ **Fix deprecated LangChain imports → langchain-qdrant**
- ✅ **Batch processing với progress tracking**
- ✅ **Auto collection naming: chunks + embedding model**
- ✅ **Collection name sanitization** (loại bỏ ký tự đặc biệt)
- ✅ **Retriever builder với 3 search modes** (similarity, MMR, score threshold)
- ✅ **Evaluation pipeline với 2 modes** (full pipeline, from collection)
- ✅ **Subprocess orchestration** để kết nối các modules
- ✅ **CLI interface** với argparse cho tất cả scripts
- ✅ **2 evaluation methods**: Document-level và Chunk-level semantic
- ✅ **4 metrics**: Hit Rate, Recall, Precision, MRR
- ✅ **Gold embeddings builder** từ Qdrant vectors (mean-pooling)
- ✅ **Cosine similarity matching** cho semantic evaluation
- ✅ **Re-embedding approach** để đảm bảo consistency
- ✅ **Professional output formatting** (clean, no excessive emojis)
- ✅ **Debug statistics** để monitor evaluation process

### In Progress:
- 🔄 Implement semantic & hybrid chunking strategies
- 🔄 Code quality improvements (type hints, error handling)

### Planned:
- 📋 **Metrics visualization** (plots, comparison tables)
- 📋 **Experiment tracking** (save results to JSON/CSV)
- 📋 **Statistical significance testing** (paired t-test, bootstrap)
- 📋 **Reranker integration** (BAAI/bge-reranker-v2-m3)
- 📋 **Hybrid search** (dense + sparse/BM25)
- 📋 **NDCG metric** (Normalized Discounted Cumulative Gain)
- 📋 Add token counting & statistics
- 📋 Support multiple output formats (Parquet, CSV)
- 📋 A/B testing framework cho chunking strategies
- 📋 **Error analysis**: Analyze failed queries
- 📋 **Per-category metrics**: Breakdown by article_type