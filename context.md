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

## 🗄️ Vector Store Setup

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

## 🔀 Hybrid Search (Dense + Sparse BM25)

### **Overview**
Hybrid search kết hợp vector similarity (dense) và keyword matching (BM25 sparse) để cải thiện retrieval quality.

### **Implementation Approach**
- **Dense retrieval**: Vector embeddings (như hiện tại)
- **Sparse retrieval**: BM25 keyword search trên text content
- **Fusion**: Reciprocal Rank Fusion (RRF) để merge kết quả

### **Collection Naming Convention**

**Dense mode** (vector only):
```
chunks_recursive_380_50_baai_bge_small_en_v1_5
```

**Hybrid mode** (dense + BM25):
```
chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid
```

Pattern: `{chunks}_{embedding_model}_{mode}`

### **src/vectorstore/hybrid_retriever.py** - Hybrid Retriever Implementation

**Key Components**:
- `HybridRetriever`: LangChain BaseRetriever subclass
- `build_bm25_index()`: Build BM25 index from documents
- `_reciprocal_rank_fusion()`: RRF algorithm to merge dense + sparse results

**Parameters**:
- `alpha`: Weight for dense vs sparse (0=sparse only, 1=dense only, 0.5=equal)
- `rrf_k`: RRF constant parameter (default=60)
- `k`: Number of documents to retrieve

**RRF Formula**:
```
score(d) = alpha * (1 / (rrf_k + rank_dense(d))) + 
           (1-alpha) * (1 / (rrf_k + rank_sparse(d)))
```

### **Updated Scripts**

#### **build_vectordb.py**
```bash
# Dense indexing (default)
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_recursive_380_50.jsonl \
  --mode dense

# Hybrid indexing (adds _hybrid suffix)
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_recursive_380_50.jsonl \
  --mode hybrid
```

**Note**: Hybrid indexing currently creates dense-only vectors with `_hybrid` suffix. Full sparse vector support coming soon.

#### **build_retriever.py**
```bash
# Dense retrieval (auto-detect from collection name)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --query "How to create events?"

# Hybrid retrieval (auto-detect from _hybrid suffix)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --query "How to create events?" \
  --alpha 0.5 \
  --rrf_k 60

# Explicit mode (override auto-detection)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --mode hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl
```

**Auto-detection Logic**:
- Collection ending with `_hybrid` → hybrid mode
- Otherwise → dense mode

**Hybrid Mode Requirements**:
- `--chunks_file` must be provided (for BM25 index)
- Same JSONL file used during indexing

#### **evaluate.py**
```bash
# Full pipeline with hybrid search
uv run python src/evaluate.py \
  --mode full \
  --strategy recursive \
  --chunk_size 380 \
  --overlap 50 \
  --retrieval_mode hybrid \
  --alpha 0.5 \
  --max_queries 100

# Evaluate existing hybrid collection
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --alpha 0.5 \
  --eval_method both

# Auto-detect chunks_file (for hybrid collections)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --eval_method both
  # Auto-detects: data/chunks/chunks_recursive_380_50.jsonl
```

**New Parameters**:
- `--retrieval_mode`: `dense` or `hybrid` (default: `dense`)
- `--chunks_file`: Path to JSONL chunks (auto-detected for hybrid if not provided)
- `--alpha`: Hybrid weight (default: 0.5)
- `--rrf_k`: RRF parameter (default: 60)

**Auto-detection**:
- Detects mode from collection name (`_hybrid` suffix)
- Infers chunks_file from collection name pattern:
  - `chunks_{strategy}_{size}_{overlap}_{embedding}_hybrid`
  - → `data/chunks/chunks_{strategy}_{size}_{overlap}.jsonl`

### **Usage Examples**

**Example 1: Full Pipeline Dense vs Hybrid**
```bash
# Dense (baseline)
uv run python src/evaluate.py \
  --mode full --strategy recursive --chunk_size 380 --overlap 50 \
  --retrieval_mode dense --max_queries 200 --eval_method both

# Hybrid (BM25 + dense)
uv run python src/evaluate.py \
  --mode full --strategy recursive --chunk_size 380 --overlap 50 \
  --retrieval_mode hybrid --alpha 0.5 --max_queries 200 --eval_method both
```

**Example 2: Evaluate Existing Collections**
```bash
# Evaluate dense collection
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --eval_method both --max_queries 200

# Evaluate hybrid collection (auto-detect chunks_file)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --eval_method both --max_queries 200
```

**Example 3: Hyperparameter Tuning**
```bash
# Test different alpha values (dense vs sparse weight)
for alpha in 0.0 0.25 0.5 0.75 1.0; do
  uv run python src/evaluate.py \
    --mode from_collection \
    --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
    --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
    --alpha $alpha \
    --eval_method both \
    --max_queries 200
done
```

### **Dependencies**
```toml
# Added to pyproject.toml
dependencies = [
    ...
    "rank-bm25>=0.2.2",  # BM25 implementation
]
```

Install:
```bash
uv pip install rank-bm25
```

### **API Usage in Code**
```python
from vectorstore.build_retriever import build_retriever

# Dense retriever (default)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    search_kwargs={"k": 5}
)

# Hybrid retriever (explicit)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid",
    mode="hybrid",
    chunks_file="data/chunks/chunks_recursive_380_50.jsonl",
    alpha=0.5,  # Equal weight
    rrf_k=60
)

# Hybrid retriever (auto-detect from _hybrid suffix)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid",
    chunks_file="data/chunks/chunks_recursive_380_50.jsonl"
)

# Use retriever
results = retriever.invoke("How to create a Wix event?")
```

---

##  Development Log

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
- ✅ **Hybrid Search Implementation** (Dense + BM25 with RRF)
- ✅ **Auto-detection of retrieval mode** from collection name
- ✅ **Auto-inference of chunks_file** from collection naming pattern
- ✅ **Collection naming convention** với `_hybrid` suffix
- ✅ **Hybrid retriever với configurable alpha & rrf_k**
- ✅ **Experiment tracking**: Auto-save results to CSV + JSON config
- ✅ **Analysis tools**: Script để compare và analyze evaluation results
- ✅ **Automation scripts**: `run_evaluate.py` (Python) và `run_evaluate.sh` (Bash)
- ✅ **Dynamic collection naming**: Bash script calls Python function để tạo collection names chính xác
- ✅ **Flexible evaluation parameters**: Configurable chunk_size, overlap, strategy via CLI

### In Progress:
- 🔄 Implement semantic & hybrid chunking strategies
- 🔄 Code quality improvements (type hints, error handling)
- 🔄 **Compare dense vs hybrid performance metrics**

### Planned:
- 📋 **Metrics visualization** (plots, comparison tables)
- 📋 **Statistical significance testing** (paired t-test, bootstrap)
- 📋 **Reranker integration** (BAAI/bge-reranker-v2-m3)
- 📋 **Full sparse vector support** in Qdrant (SPLADE embeddings)
- 📋 Add token counting & statistics
- 📋 Support multiple output formats (Parquet)
- 📋 A/B testing framework cho chunking strategies
- 📋 **Error analysis**: Analyze failed queries
- 📋 **Per-category metrics**: Breakdown by article_type

---

## 📊 Evaluation Results Tracking

### **New Feature**: Auto-save Results (Nov 9, 2025)

`src/evaluate.py` now automatically saves all evaluation results to CSV and JSON for tracking experiments.

### Files Structure

```
data/evaluate_results/
├── README.md                          # Full documentation
├── QUICKSTART.md                      # Quick start guide
├── analyze_results.py                 # Analysis script
├── evaluation_results.csv             # All results (append-only)
├── exp_<timestamp>_config.json        # Per-experiment configs
└── summary.json                       # Latest analysis summary
```

### Auto-saved Data

**CSV** (`evaluation_results.csv`):
- Append-only file với tất cả experiments
- Columns: timestamp, experiment_id, config params, metrics
- Format: Compatible với pandas, Excel

**JSON** (`exp_<timestamp>_config.json`):
- Full configuration cho mỗi experiment
- Reproducibility: All params saved
- Format: Human-readable JSON

### Usage

```bash
# Run evaluation (auto-saves results)
uv run src/evaluate.py --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --top_k 5

# Analyze all results
uv run data/evaluate_results/analyze_results.py
```

### Saved Metrics

**Document-level**: hit_rate@K, recall@K, precision@K, mrr@K, ndcg@K, coverage@K

**Semantic-level**: hit_rate@K_semantic, recall@K_semantic, precision@K_semantic, mrr@K_semantic, ndcg@K_semantic, coverage@K_semantic

### Saved Config

- Pipeline: mode, collection_name, strategy, chunk_size, overlap, embedding_model, qdrant_path, retrieval_mode
- Retriever: top_k, search_type, alpha, rrf_k
- Evaluation: eval_method, num_queries, max_queries, retrieve_k, agg_mode, cosine_threshold

### Analysis

**Python**:
```python
import pandas as pd
df = pd.read_csv('data/evaluate_results/evaluation_results.csv')

# Compare configs
df.groupby('chunk_size')[['hit_rate@5', 'mrr@5']].mean()

# Find best
best = df.loc[df['hit_rate@5'].idxmax()]
```

**Script**:
```bash
uv run data/evaluate_results/analyze_results.py
# Shows: latest experiment, comparisons, top configs, summary
```

See `data/evaluate_results/README.md` for full documentation.

---

## 🚀 Automation Scripts

### **run_evaluate.py** - Python Automation Script

Full automation script để chạy toàn bộ evaluation pipeline với Python.

**Features**:
- ✅ **Per-model evaluation**: Build → Evaluate Dense → Evaluate Hybrid (per model)
- ✅ **Lower memory**: Chỉ load 1 model tại 1 thời điểm
- ✅ **Incremental saves**: Kết quả được lưu ngay sau mỗi evaluation
- ✅ **Dynamic collection naming**: Sử dụng exact logic từ `build_vectordb.py`
- ✅ **Summary report**: Bảng tổng kết cuối cùng
- ✅ **Cross-platform**: Chạy trên Windows/Linux/Mac

**Configuration** (in file):
```python
# Models to evaluate
EMBEDDING_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
]

# Evaluation settings
EVAL_CONFIG = {
    "top_k": 5,
    "max_queries": 50,
    "retrieve_k": 50,
    "agg_mode": "max",
    "cosine_threshold": 0.75,
    "eval_method": "both",
    "alpha": 0.7,
    "rrf_k": 60,
}
```

**Usage**:
```bash
# Run full pipeline
uv run run_evaluate.py
```

**Flow**:
```
1. Chunking (optional, skip if exists)

2-3. FOR EACH model:
   Build VectorStore
   → Evaluate Dense mode (save)
   → Evaluate Hybrid mode (save)
   → Next model

4. Summary report
```

---

### **run_evaluate.sh** - Bash Automation Script

Flexible bash script với dynamic collection naming và configurable parameters.

**Features**:
- ✅ **Dynamic collection naming**: Gọi Python function `get_collection_name()` để tạo tên chính xác
- ✅ **Flexible parameters**: Configurable strategy, chunk_size, overlap via CLI
- ✅ **Per-model evaluation**: Build → Dense → Hybrid (per model)
- ✅ **No hardcoding**: Không fix cứng collection names
- ✅ **100% accuracy**: Collection names khớp hoàn toàn với `build_vectordb.py` logic

**Parameters**:
- `--strategy` (required): `recursive` hoặc `token`
- `--chunk_size` (optional, default: 380): Kích thước chunk
- `--overlap` (optional, default: 50): Overlap size

**Usage**:
```bash
# Default (chunk_size=380, overlap=50)
bash run_evaluate.sh --strategy recursive

# Custom chunk size
bash run_evaluate.sh --strategy token --chunk_size 512

# Full custom
bash run_evaluate.sh --strategy recursive --chunk_size 512 --overlap 64
```

**Dynamic Collection Naming**:
```bash
# Script calls Python function
QWEN_COLLECTION=$(get_collection_name "Qwen/Qwen3-Embedding-0.6B")
# Returns: chunks_recursive_380_50_qwen_qwen3_embedding_0_6b

# Helper function definition
get_collection_name() {
    local embedding_model=$1
    python -c "from run_evaluate import get_collection_name; \
               print(get_collection_name('${CHUNKS_FILE}', '${embedding_model}'))"
}
```

**Example Outputs**:
```bash
# --strategy recursive --chunk_size 380 --overlap 50
CHUNKS_FILE: data/chunks/chunks_recursive_380_50.jsonl
QWEN_COLLECTION: chunks_recursive_380_50_qwen_qwen3_embedding_0_6b
BGE_COLLECTION: chunks_recursive_380_50_baai_bge_m3

# --strategy token --chunk_size 512 --overlap 64
CHUNKS_FILE: data/chunks/chunks_token_512_64.jsonl
QWEN_COLLECTION: chunks_token_512_64_qwen_qwen3_embedding_0_6b
BGE_COLLECTION: chunks_token_512_64_baai_bge_m3
```

**Flow**:
```
1. Parse CLI arguments (strategy, chunk_size, overlap)
2. Build chunks filename
3. FOR EACH model:
   - Get collection name from Python
   - Build VectorStore with chunks file
   - Evaluate Dense mode
   - Evaluate Hybrid mode
4. Summary
```

**Models evaluated** (in script):
1. `Qwen/Qwen3-Embedding-0.6B`
2. `BAAI/bge-m3`

**Evaluation config** (fixed in script):
```bash
--top_k 5
--retrieve_k 20
--agg_mode max
--cosine_threshold 0.70
--eval_method both
--alpha 0.7          # Hybrid mode
--rrf_k 60           # Hybrid mode
```

---

### **Comparison: Python vs Bash Script**

| Feature | `run_evaluate.py` | `run_evaluate.sh` |
|---------|-------------------|-------------------|
| **Platform** | ✅ Windows/Linux/Mac | ⚠️ Bash (Git Bash on Windows) |
| **Collection naming** | ✅ Auto from Python | ✅ Auto from Python (calls function) |
| **Parameters** | 🔧 Edit in file | ✅ CLI arguments |
| **Models** | 🔧 3 models (edit list) | 🔧 2 models (fixed in script) |
| **Error handling** | ✅ Advanced try-catch | ⚠️ Basic |
| **Summary report** | ✅ Detailed table | ✅ Simple output |
| **Flexibility** | ⚠️ Need edit file | ✅ CLI args (strategy, size, overlap) |
| **Use case** | Multiple models, complex flow | Quick testing, different configs |

**Recommendation**:
- **Python script** (`run_evaluate.py`): Khi cần evaluate nhiều models, production runs
- **Bash script** (`run_evaluate.sh`): Khi cần test nhanh với different chunk configs

---

### **Related Documentation**

- `RUN_EVALUATE_GUIDE.md` - Python script detailed guide
- `RUN_EVALUATE_SH_GUIDE.md` - Bash script detailed guide  
- `FLOW_COMPARISON.md` - Visual comparison Bash vs Python flow
- `SCRIPT_FIX_SUMMARY.md` - Script debugging history