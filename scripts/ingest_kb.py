import argparse
from typing import Iterable, List

from tqdm import tqdm

import sys

sys.path.append("/content/rag_wixqa/src")
from rag_wixqa.data.load_wixqa import load_kb_docs
from rag_wixqa.chunking.strategies import CHUNKERS, ChunkMethod, Chunk
from rag_wixqa.vectorstores.chroma_store import build_chroma_from_chunks
from rag_wixqa.config import emb_cfg


def _batched(seq: List, batch_size: int) -> Iterable[List]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


def ingest(
    method: ChunkMethod,
    embedding_model_name: str | None = None,
    chunk_batch_size: int = 128,
    chroma_batch_size: int = 200,
):
    print("Loading KB docs...")
    kb_docs = load_kb_docs()
    total_docs = len(kb_docs)
    print(f"Loaded {total_docs} documents.")

    print(f"Chunking with method={method} (batch size={chunk_batch_size})...")
    chunker = CHUNKERS[method]
    chunks: List[Chunk] = []

    with tqdm(total=total_docs, desc="Chunking docs", unit="doc") as pbar:
        for batch in _batched(kb_docs, chunk_batch_size):
            batch_chunks = chunker(batch)
            chunks.extend(batch_chunks)
            pbar.update(len(batch))

    print(f"Total chunks generated: {len(chunks)}")

    coll_name = f"wixqa_{method}_chunks"
    print(f"Building Chroma collection: {coll_name}")
    build_chroma_from_chunks(
        chunks,
        collection_name=coll_name,
        embedding_model_name=embedding_model_name,
        chroma_batch_size=chroma_batch_size,
        reset_collection=True,
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
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=128,
        help="Number of KB docs to chunk at a time.",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=200,
        help="Number of chunks to send to Chroma per indexing batch.",
    )
    args = parser.parse_args()

    print(
        f"Using embedding model '{args.embedding_model_name}' "
        f"(dim={args.embedding_dim})"
    )
    ingest(
        args.method,  # type: ignore[arg-type]
        embedding_model_name=args.embedding_model_name,
        chunk_batch_size=args.chunk_batch_size,
        chroma_batch_size=args.chroma_batch_size,
    )
