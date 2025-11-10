# Project Context: CTAI_GenAI - WixQA Chunking Pipeline

> **QUAN TRỌNG - QUY TẮC CẬP NHẬT CONTEXT:**
> 
> Mỗi lần cập nhật file context.md này, PHẢI tuân thủ các quy tắc sau:
> 
> 1. **KHÔNG sử dụng emoji/icon** - Giữ văn bản thuần túy, chuyên nghiệp
>    - NGOẠI LỆ: Chỉ được dùng ✅ 🔄 📋 trong phần Development Log để đánh dấu trạng thái
> 
> 2. **Cấu trúc rõ ràng và nhất quán**
>    - Sử dụng heading levels (##, ###, ####) đúng thứ bậc
>    - Một phần không nên quá dài (max ~100 lines)
> 
> 3. **Nội dung súc tích, tránh lặp**
>    - Mỗi thông tin chỉ ghi một lần
>    - Xóa các ví dụ dài dòng, chỉ giữ ví dụ đại diện nhất
> 
> 4. **Code examples tối thiểu**
>    - Mỗi script/tool chỉ 1-2 ví dụ quan trọng nhất
>    - Không duplicate usage examples
> 
> 5. **Cập nhật theo nguyên tắc thời gian**
>    - Thông tin mới nhất luôn ở cuối cùng
>    - Xóa thông tin lỗi thời (deprecated)
> 
> 6. **Chất lượng > Số lượng**
>    - Giữ file dưới 1000 lines nếu có thể
>    - Tập trung vào thông tin thiết yếu để làm việc

---

## Overview
Dự án xây dựng pipeline để chunk WixQA corpus thành các đoạn văn bản nhỏ hơn phục vụ cho RAG (Retrieval-Augmented Generation) hoặc các tác vụ NLP khác.

---

## Cấu trúc Project

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

## Data Output

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

## Tech Stack
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

## Notes & TODOs

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

## Vector Store Setup

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
# Basic (auto collection name)
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_recursive_380_50.jsonl

# Custom embedding model
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_token_380_50.jsonl \
  --embedding_model "BAAI/bge-base-en-v1.5" \
  --recreate
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

## Retriever Setup

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
# Test retriever
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --query "How to create an event in Wix?" \
  --top_k 5

# MMR search
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --search_type mmr
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

retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    search_type="similarity",  # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5}
)
docs = retriever.invoke("How to create events?")
```

**Search Types**:
- **`similarity`**: Vector cosine similarity (default) - tìm docs gần nhất
- **`mmr`**: Maximal Marginal Relevance - cân bằng giữa relevance và diversity
- **`similarity_score_threshold`**: Chỉ trả về docs có score >= threshold

---

## Evaluation Pipeline

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
# Full pipeline
uv run python src/evaluate.py \
  --mode full \
  --strategy recursive \
  --chunk_size 380 \
  --overlap 50 \
  --eval_method both \
  --max_queries 50

# From existing collection
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --eval_method both \
  --top_k 5

# With custom parameters
uv run src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --top_k 5 \
  --retrieve_k 20 \
  --agg_mode max \
  --cosine_threshold 0.70 \
  --eval_method both
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
DOCUMENT-LEVEL RESULTS:
  hit_rate@5: 0.6400
  recall@5: 0.5867
  precision@5: 0.1360
  mrr@5: 0.4567

CHUNK-LEVEL SEMANTIC RESULTS:
  hit_rate@5_semantic: 1.0000
  recall@5_semantic: 0.5867
  precision@5_semantic: 0.9560
  mrr@5_semantic: 0.9567
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

## Hybrid Search (Dense + Sparse BM25)

### **Overview**
Hybrid search kết hợp vector similarity (dense) và keyword matching (BM25 sparse) để cải thiện retrieval quality.

Dự án hỗ trợ **2 implementation approaches**:

1. **Native Hybrid (Qdrant Internal)** - **RECOMMENDED**
   - Qdrant handle cả dense và sparse vectors internally
   - Sử dụng `FastEmbedSparse` với BM25 model từ Qdrant
   - Sparse vectors được index cùng với dense vectors
   - Fusion được thực hiện bởi Qdrant engine
   - **Advantages**: Faster, more efficient, no external BM25 index
   
2. **Custom Hybrid (External BM25 + RRF)** - **LEGACY**
   - External BM25 index sử dụng `rank-bm25` library
   - Manual Reciprocal Rank Fusion (RRF) để merge results
   - Requires chunks JSONL file at query time
   - **Use case**: Research purposes, custom fusion logic

### **Collection Naming Convention**

**Dense mode** (vector only):
```
chunks_recursive_380_50_baai_bge_small_en_v1_5
```

**Native Hybrid mode** (dense + sparse BM25 in Qdrant):
```
chunks_recursive_380_50_baai_bge_small_en_v1_5_native_hybrid
```

**Custom Hybrid mode** (external BM25 + RRF):
```
chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid
```

Pattern: `{chunks}_{embedding_model}_{mode}`
- No suffix = Dense
- `_native_hybrid` = Native Hybrid (Qdrant internal)
- `_hybrid` = Custom Hybrid (external BM25)

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

# Native Hybrid indexing (Qdrant internal dense + sparse BM25)
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_recursive_380_50.jsonl \
  --mode native_hybrid

# Custom Hybrid indexing (for legacy BM25+RRF)
# Note: This only creates dense vectors, BM25 index is built at query time
uv run src/vectorstore/build_vectordb.py \
  --chunks data/chunks/chunks_recursive_380_50.jsonl \
  --mode dense
```

**Key Features**:
- `--mode dense`: Vector-only indexing (default)
- `--mode native_hybrid`: Creates collection with both dense and sparse vectors
  - Automatically detects vector size from embedding model
  - Sparse vectors indexed with Qdrant BM25 model
  - Collection config: `vectors_config={"dense": VectorParams(...)}`, `sparse_vectors_config={"sparse": SparseVectorParams(...)}`

#### **build_retriever.py**
```bash
# Dense retrieval
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --query "How to create events?"

# Native Hybrid retrieval (auto-detect, no chunks_file needed)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_native_hybrid \
  --query "How to create events?"

# Custom Hybrid retrieval (requires chunks_file)
uv run python src/vectorstore/build_retriever.py \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --alpha 0.5
```

**Auto-detection Logic**:
- Collection ending with `_native_hybrid` → native hybrid mode
- Collection ending with `_hybrid` → custom hybrid mode
- Otherwise → dense mode

**Mode Requirements**:
- **Native Hybrid**: No extra requirements (sparse vectors already in Qdrant)
- **Custom Hybrid**: Requires `--chunks_file` for BM25 index

#### **evaluate.py**
```bash
# Dense evaluation
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --retrieval_mode dense \
  --eval_method both

# Native Hybrid evaluation (auto-detect)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_native_hybrid \
  --retrieval_mode native_hybrid \
  --eval_method both

# Custom Hybrid evaluation (requires chunks_file)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid \
  --retrieval_mode hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --alpha 0.5
```

**Retrieval Mode Parameters**:
- `--retrieval_mode`: `dense`, `native_hybrid`, or `hybrid` (default: `dense`)
- `--chunks_file`: Path to JSONL chunks (only for custom hybrid mode)
- `--alpha`: Custom hybrid weight (default: 0.5, not used for native_hybrid)
- `--rrf_k`: RRF parameter (default: 60, not used for native_hybrid)

**Auto-detection**:
- Detects mode from collection name suffix
- Auto-infers chunks_file for custom hybrid (if not provided)

### **Usage Examples**

```bash
# Dense retrieval
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --retrieval_mode dense \
  --eval_method both

# Native Hybrid retrieval (RECOMMENDED)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5_native_hybrid \
  --retrieval_mode native_hybrid \
  --eval_method both

# Custom Hybrid retrieval (legacy)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --retrieval_mode hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --eval_method both

# Hyperparameter tuning (custom hybrid only)
uv run python src/evaluate.py \
  --mode from_collection \
  --collection chunks_recursive_380_50_baai_bge_small_en_v1_5 \
  --retrieval_mode hybrid \
  --chunks_file data/chunks/chunks_recursive_380_50.jsonl \
  --alpha 0.7 \
  --eval_method both
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

# Dense retriever
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    search_kwargs={"k": 5}
)

# Native Hybrid retriever (RECOMMENDED)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_native_hybrid",
    mode="native_hybrid",  # or None for auto-detect
    search_kwargs={"k": 5}
)

# Custom Hybrid retriever (legacy)
retriever = build_retriever(
    collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
    mode="hybrid",
    chunks_file="data/chunks/chunks_recursive_380_50.jsonl",
    alpha=0.5
)

results = retriever.invoke("How to create a Wix event?")
```

### **Native Hybrid vs Custom Hybrid Comparison**

| Feature | Native Hybrid | Custom Hybrid |
|---------|--------------|---------------|
| **Implementation** | Qdrant internal | External BM25 + RRF |
| **Sparse vectors** | Indexed in Qdrant | Built at query time |
| **Chunks file needed** | ❌ No | ✅ Yes (at query time) |
| **Performance** | ⚡ Faster | 🐢 Slower |
| **Memory** | 💾 More efficient | 📦 Loads BM25 index |
| **Fusion algorithm** | Qdrant internal | Manual RRF |
| **Tuning** | Limited | Full control (alpha, rrf_k) |
| **Collection suffix** | `_native_hybrid` | `_hybrid` |
| **Use case** | **Production (recommended)** | Research, custom fusion |

**When to use Native Hybrid**:
- Production deployments
- When query latency matters
- When you want simplicity
- Default choice for most use cases

**When to use Custom Hybrid**:
- Research experiments
- When you need custom fusion logic
- When you want full control over RRF parameters
- A/B testing different fusion approaches

---

## Development Log

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
- ✅ **Native Hybrid Search Implementation** (Qdrant internal dense + sparse BM25)
- ✅ **Custom Hybrid Search Implementation** (External BM25 + RRF fusion)
- ✅ **Auto-detection of retrieval mode** from collection name (dense, native_hybrid, hybrid)
- ✅ **Auto-inference of chunks_file** from collection naming pattern
- ✅ **Collection naming convention** với `_native_hybrid` và `_hybrid` suffix
- ✅ **FastEmbedSparse integration** với Qdrant BM25 model
- ✅ **Dynamic vector size detection** từ embedding model
- ✅ **Dual sparse vector configs** in collection creation
- ✅ **Experiment tracking**: Auto-save results to CSV + JSON config
- ✅ **Analysis tools**: Script để compare và analyze evaluation results
- ✅ **Automation scripts**: `run_evaluate_v2.sh` (advanced, JSON config), `run_evaluate.sh` (legacy)
- ✅ **Dynamic collection naming**: Scripts call Python function để tạo collection names chính xác
- ✅ **Flexible evaluation parameters**: Configurable via JSON config hoặc CLI args
- ✅ **3-mode evaluation**: Dense, Native Hybrid, Custom Hybrid per model
- ✅ **Mock mode**: Fast testing với limited data (indexing + eval only)
- ✅ **Smart skip logic**: Auto skip chunking/indexing nếu đã tồn tại
- ✅ **Experiments config**: JSON-based configuration cho unlimited experiments
- ✅ **Progress tracking**: Color-coded SKIP/EXECUTED status
- ✅ **Suppress warnings**: Python warnings suppressed để output clean

### In Progress:
- 🔄 Implement semantic chunking strategies
- 🔄 Code quality improvements (type hints, error handling)
- 🔄 **Compare dense vs native hybrid vs custom hybrid performance metrics**

### Planned:
- 📋 **Metrics visualization** (plots, comparison tables)
- 📋 **Statistical significance testing** (paired t-test, bootstrap)
- 📋 **Reranker integration** (BAAI/bge-reranker-v2-m3)
- 📋 **SPLADE embeddings support** (alternative sparse embedding model)
- 📋 Add token counting & statistics
- 📋 Support multiple output formats (Parquet)
- 📋 A/B testing framework cho chunking strategies
- 📋 **Error analysis**: Analyze failed queries
- 📋 **Per-category metrics**: Breakdown by article_type

---

## Experiments Configuration

### **experiments.jsonl** - Experiments Config File

JSON array chứa các experiments để chạy với `run_evaluate_v2.sh`.

**Format**:
```json
[
    {
        "strategy": "recursive",
        "chunk_size": 380,
        "overlap": 50,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "mode": "dense",
        "top_k": 5,
        "retrieve_k": 20,
        "agg_mode": "max",
        "cosine_threshold": 0.70,
        "eval_method": "both",
        "alpha": 0.7,
        "rrf_k": 60
    }
]
```

**Fields**:
- `strategy` (required): Chunking strategy - `recursive`, `token`
- `chunk_size` (required): Chunk size
- `overlap` (required): Overlap size
- `embedding_model` (required): HuggingFace model name
- `mode` (required): Retrieval mode - `dense`, `native_hybrid`, `hybrid`
- `top_k` (optional, default=5): Number of docs to retrieve
- `retrieve_k` (optional, default=20): Number of chunks for doc-level eval
- `agg_mode` (optional, default=max): Aggregation mode - `max`, `sum`, `mean`
- `cosine_threshold` (optional, default=0.70): Threshold for semantic eval
- `eval_method` (optional, default=both): `document`, `semantic`, `both`
- `alpha` (optional, default=0.7): Custom hybrid weight (only for mode=hybrid)
- `rrf_k` (optional, default=60): RRF parameter (only for mode=hybrid)

**Usage Examples**:

**Grid Search Chunk Sizes**:
```json
[
    {"strategy": "recursive", "chunk_size": 256, "overlap": 32, "embedding_model": "BAAI/bge-small-en-v1.5", "mode": "dense"},
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-small-en-v1.5", "mode": "dense"},
    {"strategy": "recursive", "chunk_size": 512, "overlap": 64, "embedding_model": "BAAI/bge-small-en-v1.5", "mode": "dense"}
]
```

**Compare Models**:
```json
[
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-small-en-v1.5", "mode": "native_hybrid"},
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-m3", "mode": "native_hybrid"},
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "mode": "native_hybrid"}
]
```

**Compare Retrieval Modes**:
```json
[
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-m3", "mode": "dense"},
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-m3", "mode": "native_hybrid"},
    {"strategy": "recursive", "chunk_size": 380, "overlap": 50, "embedding_model": "BAAI/bge-m3", "mode": "hybrid", "alpha": 0.7}
]
```

---

## Evaluation Results Tracking

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

## Automation Scripts

### **run_evaluate_v2.sh** - Advanced Evaluation Pipeline (RECOMMENDED)

Script đánh giá nâng cao với experiments config và smart skip logic.

**Features**:
- ✅ **Experiments config**: Đọc từ `experiments.jsonl` (JSON array)
- ✅ **Mock mode**: Test nhanh với limited data
- ✅ **Smart skip logic**: Auto skip chunking/indexing nếu đã tồn tại
- ✅ **Colored output**: Dễ đọc với terminal colors
- ✅ **Progress tracking**: SKIP/EXECUTED status cho mỗi step
- ✅ **Python helpers**: Parse JSON và generate collection names

**Configuration** (experiments.jsonl):
```json
[
    {
        "strategy": "recursive",
        "chunk_size": 380,
        "overlap": 50,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "mode": "dense",
        "top_k": 5,
        "retrieve_k": 20,
        "agg_mode": "max",
        "cosine_threshold": 0.70,
        "eval_method": "both"
    }
]
```

**Usage**:
```bash
# Test nhanh với mock mode
bash run_evaluate_v2.sh --mock

# Full production run
bash run_evaluate_v2.sh --experiments experiments.jsonl

# Custom Qdrant path
bash run_evaluate_v2.sh --qdrant_path custom_qdrant
```

**Mock Mode Behavior**:
- Chunking: **Full data** (no limit, reusable)
- Indexing: 100 docs + `--recreate`
- Evaluation: 10 queries

**Smart Skip Logic**:
- Chunking: Skip if chunks file exists (unless mock)
- Indexing: Skip if collection exists (unless mock)
- Evaluation: Always run (để có kết quả mới)

**Flow**:
```
1. Parse experiments.jsonl
2. FOR EACH experiment:
   ├─ Check chunks exist → Skip or Run
   ├─ Check collection exist → Skip or Run
   └─ Always run evaluation
3. Summary report
```

---

### **run_evaluate.sh** - Bash Automation Script (LEGACY)

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
Script calls Python function to generate exact collection names from chunks file + embedding model.

**Example**:
```bash
bash run_evaluate.sh --strategy recursive --chunk_size 380 --overlap 50
# Generates: chunks_recursive_380_50_qwen_qwen3_embedding_0_6b
```

**Models**: `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`

**Config**: top_k=5, retrieve_k=20, agg_mode=max, cosine_threshold=0.70, alpha=0.7, rrf_k=60

---

### **Comparison: Automation Scripts**

| Feature | `run_evaluate_v2.sh` (NEW) | `run_evaluate.sh` (LEGACY) |
|---------|---------------------------|----------------------------|
| **Config source** | ✅ JSON file | 🔧 Hardcoded in script |
| **Mock mode** | ✅ Yes (`--mock`) | ❌ No |
| **Smart skip** | ✅ Auto skip if exists | ❌ No |
| **Experiments** | ✅ Unlimited (JSON array) | 🔧 Fixed 2 models |
| **Collection naming** | ✅ Auto from Python | ✅ Auto from Python |
| **Parameters** | ✅ JSON config | ✅ CLI args |
| **Progress tracking** | ✅ SKIP/EXECUTED status | ⚠️ Basic |
| **Colored output** | ✅ Yes | ❌ No |
| **Error handling** | ✅ `set -e` | ✅ Basic |
| **Flexibility** | ✅✅✅ Edit JSON only | ⚠️ Edit script |
| **Grid search** | ✅ JSON array | ❌ Manual |
| **Use case** | **Production (recommended)** | Legacy/Simple cases |

**Recommendation**:
- **run_evaluate_v2.sh** (NEW): Production, grid search, multiple experiments - **RECOMMENDED**
- **run_evaluate.sh** (LEGACY): Simple cases, backwards compatibility

---

### **Quick Start Guide**

**Step 1**: Create `experiments.jsonl`
```json
[
    {
        "strategy": "recursive",
        "chunk_size": 380,
        "overlap": 50,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "mode": "native_hybrid",
        "eval_method": "both"
    }
]
```

**Step 2**: Test với mock mode
```bash
bash run_evaluate_v2.sh --mock
# Fast test: Full chunking, 100 docs, 10 queries
```

**Step 3**: Run full evaluation
```bash
bash run_evaluate_v2.sh
# Production run: Smart skip logic, full data
```

**Step 4**: View results
```bash
cat data/evaluate_results/evaluation_results.csv
```

**Tips**:
- Mock mode để test nhanh (10 queries)
- Smart skip: Chỉ chạy lại evaluation nếu chunks/collection đã có
- JSON config: Dễ dàng grid search và A/B testing