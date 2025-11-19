from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass
from tqdm import tqdm

from langchain_core.vectorstores import VectorStoreRetriever

from ..data.models import QAPair
from ..chunking.strategies import ChunkMethod
from ..vectorstores.chroma_store import load_chroma_collection


@dataclass
class RetrievalMetrics:
    method: ChunkMethod
    recall_at_k: float
    k: int
    num_queries: int


def _build_retriever(collection_name: str, k: int) -> VectorStoreRetriever:
    vs = load_chroma_collection(collection_name)
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def evaluate_retrieval_for_method(
    qa_pairs: List[QAPair],
    collection_name: str,
    method: ChunkMethod,
    k: int = 5,
) -> RetrievalMetrics:
    retriever = _build_retriever(collection_name, k=k)

    hits = 0
    for qa in tqdm(qa_pairs, desc=f"Evaluating {method}"):
        if not qa.kb_ids:
            # if no ground-truth doc ids, skip this question
            continue
        docs = retriever.invoke(qa.question)
        retrieved_ids = {d.metadata.get("doc_id") for d in docs}
        if retrieved_ids & set(qa.kb_ids):
            hits += 1

    num_q = len([q for q in qa_pairs if q.kb_ids])
    recall = hits / num_q if num_q else 0.0

    return RetrievalMetrics(
        method=method,
        recall_at_k=recall,
        k=k,
        num_queries=num_q,
    )


def evaluate_all_methods(
    qa_pairs: List[QAPair],
    methods: List[ChunkMethod],
    k: int = 5,
) -> Dict[ChunkMethod, RetrievalMetrics]:
    results: Dict[ChunkMethod, RetrievalMetrics] = {}
    for m in methods:
        coll_name = f"wixqa_{m}_chunks"
        metrics = evaluate_retrieval_for_method(
            qa_pairs=qa_pairs,
            collection_name=coll_name,
            method=m,
            k=k,
        )
        results[m] = metrics
    return results
