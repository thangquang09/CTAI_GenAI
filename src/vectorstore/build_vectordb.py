import argparse
import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from tqdm import tqdm

from langchain_qdrant import QdrantVectorStore


def read_jsonl(path: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def make_documents(jsonl_path: str, limit: Optional[int] = None) -> List[Document]:
    docs: List[Document] = []
    for row in read_jsonl(jsonl_path, limit=limit):
        text = row.get("text") or ""
        metadata = {
            "chunk_id": row.get("chunk_id"),
            "article_id": row.get("article_id"),
            "title": row.get("title"),
            "url": row.get("url"),
            "article_type": row.get("article_type"),
            "position": row.get("position"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def build_qdrant_local(
    docs: List[Document],
    collection_name: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "gte-multilingual-base",
    recreate: bool = False,
    batch_size: int = 64,
    use_grpc: bool = False,
):
    """
    Build Qdrant vector store locally using the new langchain-qdrant API.
    """
    os.makedirs(qdrant_path, exist_ok=True)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model, encode_kwargs={"normalize_embeddings": True}
    )

    # Delete collection if recreate flag is set
    if recreate:
        # Use a temporary client just for deletion
        temp_client = QdrantClient(path=qdrant_path, prefer_grpc=use_grpc)
        try:
            temp_client.delete_collection(collection_name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist yet
        finally:
            # Close the client to release the lock
            temp_client.close()

    # Check if we have documents
    n = len(docs)
    if n == 0:
        print("No documents to index.")
        return

    print(
        f"Indexing {n} chunks into Qdrant collection '{collection_name}' at '{qdrant_path}'"
    )

    # Prepare IDs for documents
    # Qdrant requires UUIDs for IDs, so we generate them from chunk_id using UUID5
    # This ensures consistent UUIDs for the same chunk_id across runs
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")  # Fixed namespace
    ids = [
        str(uuid.uuid5(namespace, d.metadata.get("chunk_id") or f"{i}"))
        for i, d in enumerate(docs)
    ]

    # Create vector store and collection using from_documents
    # This will automatically create the collection with proper vector config
    vector_store = None
    with tqdm(total=n, desc="Creating collection & indexing", unit="doc") as pbar:
        # Process in batches to show progress
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_docs = docs[start:end]
            batch_ids = ids[start:end]

            if start == 0:
                # First batch: create the vector store and collection
                vector_store = QdrantVectorStore.from_documents(
                    documents=batch_docs,
                    embedding=embeddings,
                    ids=batch_ids,
                    collection_name=collection_name,
                    path=qdrant_path,
                    prefer_grpc=use_grpc,
                )
            else:
                # Subsequent batches: add to existing collection
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)

            pbar.update(end - start)

    # Print collection info using the vector_store's client
    if vector_store:
        client = vector_store.client
        info = client.get_collection(collection_name)
        print(f"✅ Collection status: {info.status}")
        print(f"✅ Vectors count: {client.count(collection_name).count}")

    return vector_store


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build local Qdrant vector DB from chunked JSONL"
    )
    ap.add_argument(
        "--chunks", type=str, required=True, help="Đường dẫn file JSONL chứa các chunk"
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="langchain_qdrant",
        help="Thư mục lưu Qdrant (on-disk)",
    )
    ap.add_argument(
        "--collection",
        type=str,
        default="",
        help="Tên collection (mặc định: lấy từ tên file chunks)",
    )
    ap.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Giới hạn số chunk để index (0 = tất cả)"
    )
    ap.add_argument("--recreate", action="store_true", help="Xóa & tạo mới collection")
    ap.add_argument(
        "--batch_size", type=int, default=256, help="Kích thước batch khi upsert"
    )
    ap.add_argument(
        "--grpc", action="store_true", help="Dùng gRPC local (thường không cần)"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    # Auto-generate collection name from chunks filename if not provided
    if not args.collection:
        chunks_basename = os.path.basename(args.chunks)
        # Remove .jsonl extension
        collection_name = chunks_basename.replace(".jsonl", "")
    else:
        collection_name = args.collection

    docs = make_documents(args.chunks, limit=limit)
    build_qdrant_local(
        docs=docs,
        collection_name=collection_name,
        qdrant_path=args.out_dir,
        embedding_model=args.embedding_model,
        recreate=args.recreate,
        batch_size=args.batch_size,
        use_grpc=args.grpc,
    )


if __name__ == "__main__":
    main()
