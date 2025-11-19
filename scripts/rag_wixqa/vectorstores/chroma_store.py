from typing import List
from chromadb.config import Settings
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..embeddings.embedding_models import get_default_embedding
from ..config import paths
from ..chunking.strategies import Chunk


def _chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    docs: List[Document] = []
    for ch in chunks:
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


def build_chroma_from_chunks(
    chunks: List[Chunk],
    collection_name: str,
) -> Chroma:
    embedding = get_default_embedding()

    client = chromadb.PersistentClient(
        path=str(paths.chroma_root),
        settings=Settings(anonymized_telemetry=False),
    )

    docs = _chunks_to_documents(chunks)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=collection_name,
        client=client,
    )
    return vectorstore


def load_chroma_collection(collection_name: str) -> Chroma:
    embedding = get_default_embedding()
    client = chromadb.PersistentClient(
        path=str(paths.chroma_root),
        settings=Settings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding,
    )
