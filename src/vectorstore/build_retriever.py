"""
Build retriever from Qdrant vector store for RAG applications.

Usage:
    # Basic retriever
    retriever = build_retriever(
        collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        qdrant_path="langchain_qdrant",
        embedding_model="BAAI/bge-small-en-v1.5"
    )
    
    # Search with default top_k=5
    docs = retriever.invoke("How to create a Wix event?")
    
    # Search with custom top_k
    docs = retriever.invoke("How to create a Wix event?", top_k=10)
    
    # With search kwargs
    retriever = build_retriever(..., search_type="mmr", search_kwargs={"k": 10})
"""

import argparse
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


def build_retriever(
    collection_name: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    search_type: str = "similarity",
    search_kwargs: Optional[Dict[str, Any]] = None,
    use_grpc: bool = False,
    device: str = "cpu",
) -> BaseRetriever:
    """
    Build a retriever from existing Qdrant collection.
    
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
        device: Device cho embedding model - "cuda", "mps", "cpu"
    
    Returns:
        BaseRetriever: LangChain retriever object có thể dùng .invoke(query)
    
    Search Types:
        - "similarity": Vector similarity search (default)
        - "mmr": Maximal Marginal Relevance (đa dạng hóa kết quả)
        - "similarity_score_threshold": Chỉ trả về docs có score >= threshold
    
    Examples:
        >>> # Basic similarity search với top_k=5
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        ...     search_kwargs={"k": 5}
        ... )
        >>> docs = retriever.invoke("How to create events in Wix?")
        
        >>> # MMR search (đa dạng hóa)
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        ...     search_type="mmr",
        ...     search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.5}
        ... )
        
        >>> # Với score threshold
        >>> retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        ...     search_type="similarity_score_threshold",
        ...     search_kwargs={"score_threshold": 0.7, "k": 10}
        ... )
    """
    # Sanitize collection name
    collection_name = sanitize_collection_name(collection_name)
    
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
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
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
    
    # Create retriever với search type và kwargs
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=default_search_kwargs,
    )
    
    print("✅ Retriever ready!")
    print(f"   Search type: {search_type}")
    print(f"   Search kwargs: {default_search_kwargs}")
    
    return retriever


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
        help="Loại search",
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


if __name__ == "__main__":
    main()

