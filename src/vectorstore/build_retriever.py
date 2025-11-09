"""
Build retriever from Qdrant vector store for RAG applications.

Supports both dense-only and hybrid (dense + BM25) retrieval modes.

Usage:
    # Dense retrieval (default)
    retriever = build_retriever(
        collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        qdrant_path="langchain_qdrant",
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    
    # Hybrid retrieval (dense + BM25)
    retriever = build_retriever(
        collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid",
        qdrant_path="langchain_qdrant",
        embedding_model="BAAI/bge-small-en-v1.5",
        mode="hybrid",
        chunks_file="data/chunks/chunks_recursive_380_50.jsonl"
    )
    
    # Search with default top_k=5
    docs = retriever.invoke("How to create a Wix event?")
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name để tránh lỗi với Qdrant.
    - Chuyển thành lowercase
    - Thay thế ký tự đặc biệt bằng underscore
    - Loại bỏ các ký tự không hợp lệ
    - Giới hạn độ dài tối đa 255 ký tự
    
    Examples:
        'BAAI/bge-small-en-v1.5' -> 'baai_bge_small_en_v1_5'
        'sentence-transformers/all-MiniLM-L6-v2' -> 'sentence_transformers_all_minilm_l6_v2'
    """
    name = name.lower()
    name = re.sub(r'[/\-.]', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > 255:
        name = name[:255]
    return name


def detect_retriever_mode(collection_name: str) -> str:
    """
    Auto-detect retriever mode from collection name.
    
    Args:
        collection_name: Collection name (may contain _hybrid suffix)
        
    Returns:
        "hybrid" if collection ends with "_hybrid", else "dense"
        
    Examples:
        >>> detect_retriever_mode("chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid")
        "hybrid"
        >>> detect_retriever_mode("chunks_recursive_380_50_baai_bge_small_en_v1_5")
        "dense"
    """
    collection_name_lower = collection_name.lower()
    if collection_name_lower.endswith("_hybrid"):
        return "hybrid"
    return "dense"


def load_documents_from_jsonl(jsonl_path: str) -> List[Document]:
    """
    Load documents from JSONL file for BM25 indexing.
    
    Args:
        jsonl_path: Path to JSONL chunks file
        
    Returns:
        List of Document objects
    """
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = row.get("text", "")
                metadata = {
                    "chunk_id": row.get("chunk_id"),
                    "article_id": row.get("article_id"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "article_type": row.get("article_type"),
                    "position": row.get("position"),
                }
                docs.append(Document(page_content=text, metadata=metadata))
    
    print(f"📚 Loaded {len(docs)} documents from {jsonl_path}")
    return docs


def build_retriever(
    collection_name: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    search_type: str = "similarity",
    search_kwargs: Optional[Dict[str, Any]] = None,
    use_grpc: bool = False,
    mode: Optional[str] = None,
    chunks_file: Optional[str] = None,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> BaseRetriever:
    """
    Build a retriever from existing Qdrant collection.
    
    Supports both dense-only and hybrid (dense + BM25) retrieval.
    
    Args:
        collection_name: Tên collection trong Qdrant (sẽ được sanitize)
        qdrant_path: Đường dẫn đến Qdrant storage directory
        embedding_model: HuggingFace embedding model (phải khớp với model đã dùng khi build)
        search_type: Loại search ("similarity", "mmr", "similarity_score_threshold")
        search_kwargs: Tham số cho search, ví dụ:
            - {"k": 5} - số documents trả về (default=4 trong LangChain)
            - {"score_threshold": 0.5} - threshold cho similarity_score_threshold
            - {"fetch_k": 20, "lambda_mult": 0.5} - cho MMR mode
        use_grpc: Sử dụng gRPC cho local client
        mode: Retrieval mode - "dense", "hybrid", hoặc None (auto-detect from collection name)
        chunks_file: Path to JSONL chunks file (required for hybrid mode)
        alpha: Hybrid mode weight (0=sparse only, 1=dense only, 0.5=equal)
        rrf_k: RRF parameter for hybrid mode (default=60)
    
    Returns:
        BaseRetriever: LangChain retriever object có thể dùng .invoke(query)
    
    Modes:
        - "dense": Vector similarity search only (default)
        - "hybrid": Dense + BM25 with Reciprocal Rank Fusion
        - None: Auto-detect from collection name (collections ending with "_hybrid" use hybrid mode)
    
    Examples:
        >>> # Dense retrieval (default)
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        ...     search_kwargs={"k": 5}
        ... )
        
        >>> # Hybrid retrieval (explicit)
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid",
        ...     mode="hybrid",
        ...     chunks_file="data/chunks/chunks_recursive_380_50.jsonl",
        ...     alpha=0.5  # Equal weight
        ... )
        
        >>> # Hybrid retrieval (auto-detect)
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5_hybrid",
        ...     chunks_file="data/chunks/chunks_recursive_380_50.jsonl"
        ... )  # Auto-detects hybrid mode from "_hybrid" suffix
    """
    # Sanitize collection name
    collection_name = sanitize_collection_name(collection_name)
    
    # Auto-detect mode if not specified
    if mode is None:
        mode = detect_retriever_mode(collection_name)
        print(f"🔍 Auto-detected mode: {mode}")
    
    # Check if collection exists
    if not os.path.exists(qdrant_path):
        raise FileNotFoundError(
            f"Qdrant path not found: {qdrant_path}. "
            "Make sure to run build_vectordb.py first."
        )
    
    # Initialize client to check collection
    client = QdrantClient(path=qdrant_path, prefer_grpc=use_grpc)
    try:
        collection_info = client.get_collection(collection_name)
        vector_count = client.count(collection_name).count
        print(f"📚 Loading collection: {collection_name}")
        print(f"   Status: {collection_info.status}")
        print(f"   Vectors: {vector_count}")
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found in {qdrant_path}. "
            f"Available collections: {[c.name for c in client.get_collections().collections]}"
        ) from e
    finally:
        client.close()
    
    # Initialize embeddings (must match the model used during indexing)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model, 
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Initialize vector store
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=qdrant_path, prefer_grpc=use_grpc),
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Default search kwargs
    default_search_kwargs = {"k": 5}  # LangChain default is 4, we use 5
    if search_kwargs:
        default_search_kwargs.update(search_kwargs)
    
    # Build dense retriever first
    dense_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=default_search_kwargs,
    )
    
    # Return hybrid or dense retriever based on mode
    if mode == "hybrid":
        # Hybrid mode requires chunks_file
        if not chunks_file:
            raise ValueError(
                "Hybrid mode requires chunks_file parameter. "
                "Please provide path to JSONL chunks file."
            )
        
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        # Load documents for BM25
        print("\n🔧 Building hybrid retriever...")
        documents = load_documents_from_jsonl(chunks_file)
        
        # Import hybrid retriever
        from .hybrid_retriever import build_hybrid_retriever
        
        # Build hybrid retriever
        retriever = build_hybrid_retriever(
            dense_retriever=dense_retriever,
            documents=documents,
            k=default_search_kwargs["k"],
            alpha=alpha,
            rrf_k=rrf_k,
        )
        
        return retriever
    else:
        # Dense mode
        print("✅ Dense retriever ready!")
        print(f"   Search type: {search_type}")
        print(f"   Search kwargs: {default_search_kwargs}")
        
        return dense_retriever


def search_documents(
    retriever: BaseRetriever,
    query: str,
    top_k: Optional[int] = None,
    **kwargs: Any
) -> List[Document]:
    """
    Helper function để search với retriever.
    
    Args:
        retriever: Retriever object từ build_retriever()
        query: Query string
        top_k: Số documents trả về (override search_kwargs["k"] nếu provided)
        **kwargs: Additional search parameters
    
    Returns:
        List[Document]: Danh sách documents với metadata
    
    Examples:
        >>> retriever = build_retriever(...)
        >>> docs = search_documents(retriever, "How to create events?", top_k=10)
        >>> for doc in docs:
        ...     print(f"Title: {doc.metadata['title']}")
        ...     print(f"Content: {doc.page_content[:100]}...")
    """
    # Update config nếu có top_k
    if top_k is not None:
        # LangChain retrievers support config override
        return retriever.invoke(query, config={"configurable": {"k": top_k}})
    else:
        return retriever.invoke(query, **kwargs)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build retriever and test search from Qdrant collection"
    )
    ap.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Tên collection (ví dụ: chunks_recursive_380_50_baai_bge_small_en_v1_5)",
    )
    ap.add_argument(
        "--qdrant_path",
        type=str,
        default="langchain_qdrant",
        help="Đường dẫn đến Qdrant storage",
    )
    ap.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model (phải khớp với khi build)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["dense", "hybrid"],
        help="Retrieval mode (None=auto-detect from collection name)",
    )
    ap.add_argument(
        "--chunks_file",
        type=str,
        default="",
        help="Path to JSONL chunks file (required for hybrid mode)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid mode weight: 0=sparse only, 1=dense only (default=0.5)",
    )
    ap.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF parameter for hybrid mode (default=60)",
    )
    ap.add_argument(
        "--query",
        type=str,
        default="",
        help="Test query (nếu muốn test search)",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Số documents trả về",
    )
    ap.add_argument(
        "--search_type",
        type=str,
        default="similarity",
        choices=["similarity", "mmr", "similarity_score_threshold"],
        help="Loại search (chỉ dùng cho dense retriever)",
    )
    ap.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Score threshold (chỉ dùng với similarity_score_threshold)",
    )
    ap.add_argument(
        "--grpc",
        action="store_true",
        help="Dùng gRPC cho local client",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Prepare search kwargs
    search_kwargs = {"k": args.top_k}
    if args.search_type == "similarity_score_threshold" and args.score_threshold:
        search_kwargs["score_threshold"] = args.score_threshold
    elif args.search_type == "mmr":
        # MMR specific params
        search_kwargs["fetch_k"] = args.top_k * 5  # Fetch more for diversity
        search_kwargs["lambda_mult"] = 0.5  # Balance relevance vs diversity
    
    # Build retriever
    retriever = build_retriever(
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
        search_type=args.search_type,
        search_kwargs=search_kwargs,
        use_grpc=args.grpc,
        mode=args.mode,
        chunks_file=args.chunks_file if args.chunks_file else None,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
    )
    
    # Test search if query provided
    if args.query:
        print(f"\n🔍 Test search: '{args.query}'")
        print("=" * 80)
        
        docs = retriever.invoke(args.query)
        
        print(f"\n📊 Found {len(docs)} documents:\n")
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc.metadata.get('title', 'N/A')}")
            print(f"    Article: {doc.metadata.get('article_id', 'N/A')}")
            print(f"    Position: {doc.metadata.get('position', 'N/A')}")
            print(f"    URL: {doc.metadata.get('url', 'N/A')}")
            print(f"    Content: {doc.page_content[:200]}...")
            print()
    else:
        print("\n💡 Tip: Dùng --query để test search")
        print("   Example: python build_retriever.py --collection <name> --query 'How to create events?'")
        print("\n💡 For hybrid mode:")
        print("   python build_retriever.py --collection <name>_hybrid --chunks_file data/chunks/<file>.jsonl --query 'test'")


if __name__ == "__main__":
    main()

