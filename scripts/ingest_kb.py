import argparse
from tqdm import tqdm
import sys
sys.path.append("/content/rag_wixqa/src")
from rag_wixqa.data.load_wixqa import load_kb_docs
from rag_wixqa.chunking.strategies import CHUNKERS, ChunkMethod
from rag_wixqa.vectorstores.chroma_store import build_chroma_from_chunks


def ingest(method: ChunkMethod):
    print(f"Loading KB docs...")
    kb_docs = load_kb_docs()

    print(f"Chunking with method={method}...")
    chunker = CHUNKERS[method]
    chunks = chunker(kb_docs)

    print(f"Total chunks: {len(chunks)}")

    coll_name = f"wixqa_{method}_chunks"
    print(f"Building Chroma collection: {coll_name}")
    build_chroma_from_chunks(chunks, collection_name=coll_name)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="recursive",
        choices=list(CHUNKERS.keys()),
    )
    args = parser.parse_args()
    ingest(args.method)  # type: ignore[arg-type]
