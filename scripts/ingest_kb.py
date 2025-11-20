import argparse
from tqdm import tqdm
import sys

sys.path.append("/content/rag_wixqa/src")
from rag_wixqa.data.load_wixqa import load_kb_docs
from rag_wixqa.chunking.strategies import CHUNKERS, ChunkMethod
from rag_wixqa.vectorstores.chroma_store import build_chroma_from_chunks
from rag_wixqa.config import emb_cfg


def ingest(method: ChunkMethod, embedding_model_name: str | None = None):
    print(f"Loading KB docs...")
    kb_docs = load_kb_docs()

    print(f"Chunking with method={method}...")
    chunker = CHUNKERS[method]
    chunks = chunker(kb_docs)

    print(f"Total chunks: {len(chunks)}")

    coll_name = f"wixqa_{method}_chunks"
    print(f"Building Chroma collection: {coll_name}")
    build_chroma_from_chunks(
        chunks,
        collection_name=coll_name,
        embedding_model_name=embedding_model_name,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="recursive",
        choices=list(CHUNKERS.keys()),
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=emb_cfg.model_name,
        help="HF embedding model to encode chunks.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=emb_cfg.dim,
        help="Embedding dimension (for logging/reference).",
    )
    args = parser.parse_args()

    print(
        f"Using embedding model '{args.embedding_model_name}' "
        f"(dim={args.embedding_dim})"
    )
    ingest(
        args.method,  # type: ignore[arg-type]
        embedding_model_name=args.embedding_model_name,
    )
