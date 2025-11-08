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
  - Standard: `argparse`, `json`, `os`, `typing`

---

## ⚠️ Notes & TODOs

### Current Issues:
1. **Type hints**: `Dict[str, any]` cần sửa thành `Dict[str, Any]` (viết hoa)
2. **Incomplete strategies**: `semantic`, `hybrid_para_token` chưa implement (chỉ có `pass`)
3. **Error handling**: Thiếu try-except cho file I/O và dataset loading
4. **Dead code**: Comment `# "n_tokens": count_tokens(p),` nên xóa hoặc implement

### Recommendations:
- Thêm logging để track progress
- Implement các strategy còn lại
- Thêm validation cho output
- Consider batch processing cho dataset lớn
- Thêm metrics: avg chunk length, token distribution

---

## 📝 Development Log

### Completed:
- ✅ Setup chunking pipeline với 2 strategies (recursive, token)
- ✅ Generate chunks cho toàn bộ WixQA corpus
- ✅ Output format chuẩn hóa với metadata đầy đủ

### In Progress:
- 🔄 Implement semantic & hybrid strategies
- 🔄 Code quality improvements (type hints, error handling)

### Planned:
- 📋 Add token counting & statistics
- 📋 Evaluation metrics cho chunk quality
- 📋 Support multiple output formats (Parquet, CSV)