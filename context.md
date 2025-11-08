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

## �📝 Development Log

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

### In Progress:
- 🔄 Implement semantic & hybrid strategies
- 🔄 Code quality improvements (type hints, error handling)

### Planned:
- 📋 Add token counting & statistics
- 📋 Evaluation metrics cho chunk quality
- 📋 Support multiple output formats (Parquet, CSV)
- 📋 RAG pipeline implementation
- 📋 Retrieval evaluation metrics