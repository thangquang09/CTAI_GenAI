from __future__ import annotations
from typing import Iterable, List, Literal, Dict, Callable
from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from sentence_transformers import SentenceTransformer
import numpy as np

from ..config import chunk_cfg
from ..data.models import KBDoc


ChunkMethod = Literal["recursive", "token", "semantic"]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    method: ChunkMethod
    order: int


def recursive_chunk(
    docs: Iterable[KBDoc],
    chunk_size: int = chunk_cfg.default_chunk_size,
    chunk_overlap: int = chunk_cfg.default_chunk_overlap,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for d in docs:
        texts = splitter.split_text(d.contents)
        for i, t in enumerate(texts):
            chunks.append(
                Chunk(
                    doc_id=d.doc_id,
                    chunk_id=f"{d.doc_id}::rec::{i}",
                    text=t,
                    method="recursive",
                    order=i,
                )
            )
    return chunks


def token_chunk(
    docs: Iterable[KBDoc],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> List[Chunk]:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Chunk] = []
    for d in docs:
        texts = splitter.split_text(d.contents)
        for i, t in enumerate(texts):
            chunks.append(
                Chunk(
                    doc_id=d.doc_id,
                    chunk_id=f"{d.doc_id}::tok::{i}",
                    text=t,
                    method="token",
                    order=i,
                )
            )
    return chunks


def semantic_chunk(
    docs: Iterable[KBDoc],
    sent_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    target_chunk_tokens: int = 220,
) -> List[Chunk]:
    """
    Very simple semantic chunking:
      - split by sentences (naively on '.')
      - embed each sentence
      - group sentences into chunks of ~target size by cosine similarity
    This is a toy example but shows the idea.
    """
    model = SentenceTransformer(sent_model_name)

    chunks: List[Chunk] = []
    for d in docs:
        # naive sentence split; you could use nltk/spacy instead
        sentences = [s.strip() for s in d.contents.split(".") if s.strip()]
        if not sentences:
            continue

        embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=32)
        current_chunk_sentences = [sentences[0]]
        current_embeds = [embeddings[0]]

        def chunk_len(sent_list: List[str]) -> int:
            # rough token proxy: 1 token ~ 0.75 word
            return sum(len(s.split()) for s in sent_list)

        def avg_vec(vecs: List[np.ndarray]) -> np.ndarray:
            return np.mean(np.vstack(vecs), axis=0)

        for s, e in zip(sentences[1:], embeddings[1:]):
            current_avg = avg_vec(current_embeds)
            cos_sim = float(
                np.dot(current_avg, e) /
                (np.linalg.norm(current_avg) * np.linalg.norm(e) + 1e-9)
            )

            if chunk_len(current_chunk_sentences + [s]) > target_chunk_tokens and cos_sim < 0.75:
                # start new chunk
                text = ". ".join(current_chunk_sentences) + "."
                idx = len(chunks)  # global order (not per-doc, fine for now)
                chunks.append(
                    Chunk(
                        doc_id=d.doc_id,
                        chunk_id=f"{d.doc_id}::sem::{idx}",
                        text=text,
                        method="semantic",
                        order=idx,
                    )
                )
                current_chunk_sentences = [s]
                current_embeds = [e]
            else:
                current_chunk_sentences.append(s)
                current_embeds.append(e)

        # flush last chunk
        if current_chunk_sentences:
            text = ". ".join(current_chunk_sentences) + "."
            idx = len(chunks)
            chunks.append(
                Chunk(
                    doc_id=d.doc_id,
                    chunk_id=f"{d.doc_id}::sem::{idx}",
                    text=text,
                    method="semantic",
                    order=idx,
                )
            )

    return chunks


CHUNKERS: Dict[ChunkMethod, Callable[..., List[Chunk]]] = {
    "recursive": recursive_chunk,
    "token": token_chunk,
    "semantic": semantic_chunk,
}
