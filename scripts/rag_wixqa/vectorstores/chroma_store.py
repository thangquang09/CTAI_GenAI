from typing import Iterable, List

import chromadb
from chromadb.config import Settings
try:
    from chromadb.errors import InvalidCollectionException
except ImportError:  # older chromadb versions renamed the exception
    InvalidCollectionException = Exception  # type: ignore[assignment]
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from ..embeddings.embedding_models import build_embedding
from ..config import paths
from ..chunking.strategies import Chunk


def _chunk_batch_to_documents(chunk_batch: Iterable[Chunk]) -> List[Document]:
    docs: List[Document] = []
    for ch in chunk_batch:
        docs.append(
            Document(
                page_content=ch.text,
                metadata={
                    "doc_id": ch.doc_id,
                    "chunk_id": ch.chunk_id,
                    "method": ch.method,
                    "order": ch.order,
                },
            )
        )
    return docs


def _batched_chunks(chunks: List[Chunk], batch_size: int) -> Iterable[List[Chunk]]:
    for idx in range(0, len(chunks), batch_size):
        yield chunks[idx : idx + batch_size]


def build_chroma_from_chunks(
    chunks: List[Chunk],
    collection_name: str,
    embedding_model_name: str | None = None,
    chroma_batch_size: int = 200,
    reset_collection: bool = True,
) -> Chroma:
    embedding = build_embedding(embedding_model_name)

    client = chromadb.PersistentClient(
        path=str(paths.chroma_root),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except InvalidCollectionException:
            pass

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding,
    )

    total_chunks = len(chunks)
    if total_chunks == 0:
        return vectorstore

    chroma_batch_size = max(1, chroma_batch_size)

    with tqdm(total=total_chunks, desc="Indexing chunks", unit="chunk") as pbar:
        for chunk_batch in _batched_chunks(chunks, chroma_batch_size):
            docs_batch = _chunk_batch_to_documents(chunk_batch)
            vectorstore.add_documents(docs_batch)
            pbar.update(len(chunk_batch))

    try:
        client.persist()
    except AttributeError:
        pass

    return vectorstore


def load_chroma_collection(
    collection_name: str,
    embedding_model_name: str | None = None,
) -> Chroma:
    embedding = build_embedding(embedding_model_name)
    client = chromadb.PersistentClient(
        path=str(paths.chroma_root),
        settings=Settings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding,
    )
