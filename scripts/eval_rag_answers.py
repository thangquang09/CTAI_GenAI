import argparse
import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from rag_wixqa.data.load_wixqa import load_qa_pairs
from rag_wixqa.data.models import QAPair
from rag_wixqa.rag.pipeline import Qwen3RAGPipeline, RAGConfig
from rag_wixqa.config import emb_cfg


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compare_answer(prediction: str, reference: str) -> Dict[str, float]:
    """
    Lightweight lexical metrics that do not require external dependencies.
    """
    pred_norm = _normalize(prediction)
    ref_norm = _normalize(reference)

    if not ref_norm:
        return {"exact_match": 0.0, "containment": 0.0, "similarity": 0.0}

    exact = 1.0 if pred_norm == ref_norm else 0.0
    containment = 1.0 if (ref_norm in pred_norm) or (pred_norm in ref_norm) else 0.0
    similarity = SequenceMatcher(None, pred_norm, ref_norm).ratio()

    return {
        "exact_match": exact,
        "containment": containment,
        "similarity": similarity,
    }


def sample_qa_pairs(qa_pairs: List[QAPair], sample_size: Optional[int]) -> List[QAPair]:
    if not sample_size or sample_size >= len(qa_pairs):
        return qa_pairs
    return random.sample(qa_pairs, sample_size)


def _compute_retrieval_metrics_at_k(
    retrieved_ids: List[str], 
    ground_truth_ids: List[str], 
    k_values: List[int]
) -> Dict[str, float]:
    """Compute retrieval metrics at different k values."""
    if not ground_truth_ids:
        return {}
    
    relevant_set = set(ground_truth_ids)
    metrics: Dict[str, float] = {}
    
    for k in k_values:
        retrieved_at_k = retrieved_ids[:k]
        hits_at_k = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_set]
        
        recall_at_k = len(hits_at_k) / len(ground_truth_ids) if ground_truth_ids else 0.0
        precision_at_k = len(hits_at_k) / k if k > 0 else 0.0
        hit_at_k = 1.0 if hits_at_k else 0.0
        
        mrr_at_k = 0.0
        for rank, doc_id in enumerate(retrieved_at_k, start=1):
            if doc_id in relevant_set:
                mrr_at_k = 1.0 / rank
                break
        
        metrics[f"recall@{k}"] = recall_at_k
        metrics[f"precision@{k}"] = precision_at_k
        metrics[f"hit_rate@{k}"] = hit_at_k
        metrics[f"mrr@{k}"] = mrr_at_k
    
    return metrics


def _batched(seq: Sequence[QAPair], batch_size: Optional[int]) -> Iterable[List[QAPair]]:
    if not batch_size or batch_size <= 0 or batch_size >= len(seq):
        yield list(seq)
        return
    for idx in range(0, len(seq), batch_size):
        yield list(seq[idx : idx + batch_size])


def evaluate_rag_answers(args: argparse.Namespace) -> Dict[str, float]:
    qa_pairs = load_qa_pairs(split=args.split)
    qa_pairs = sample_qa_pairs(qa_pairs, args.sample_size)
    total_questions = len(qa_pairs)
    
    k_values = args.k_values or [1, 3, 5, 10]

    rag_cfg = RAGConfig(
        collection_name=args.collection_name,
        model_name=args.model_name,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        embedding_model_name=args.embedding_model_name,
        embedding_dim=args.embedding_dim,
        use_reranker=(not args.disable_reranker)
        and bool(args.reranker_model_name),
        reranker_model_name=args.reranker_model_name,
        reranker_top_k=args.reranker_top_k,
        reranker_max_length=args.reranker_max_length,
        reranker_device=args.reranker_device,
    )
    rag = Qwen3RAGPipeline(rag_cfg)

    per_question_metrics: List[Dict[str, float]] = []
    per_question_retrieval_metrics: List[Dict[str, float]] = []
    retrieval_hits = 0
    retrieval_total = 0

    results_file = None
    if args.results_path:
        results_path = Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite_results else "a"
        results_file = results_path.open(mode, encoding="utf-8")

    batch_size = args.batch_size

    try:
        with tqdm(
            total=total_questions,
            desc="Evaluating RAG answers",
            unit="qa",
        ) as progress:
            for batch in _batched(qa_pairs, batch_size):
                for qa in batch:
                    result = rag.answer(
                        qa.question,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        return_docs=True,
                    )

                    answer = result["answer"]
                    metrics = compare_answer(answer, qa.answer)
                    per_question_metrics.append(metrics)

                    docs = result.get("docs") or []
                    retrieved_ids_ordered = [d.metadata.get("doc_id") for d in docs]
                    
                    # Compute retrieval metrics at k
                    retrieval_metrics = _compute_retrieval_metrics_at_k(
                        retrieved_ids_ordered, 
                        qa.kb_ids or [], 
                        k_values
                    )
                    if retrieval_metrics:
                        per_question_retrieval_metrics.append(retrieval_metrics)
                    
                    if qa.kb_ids:
                        retrieval_total += 1
                        retrieved_ids_set = {doc_id for doc_id in retrieved_ids_ordered if doc_id}
                        if retrieved_ids_set & set(qa.kb_ids):
                            retrieval_hits += 1

                    if results_file:
                        record = {
                            "question": qa.question,
                            "reference_answer": qa.answer,
                            "predicted_answer": answer,
                            "metrics": metrics,
                            "retrieval_metrics": retrieval_metrics,
                            "retrieved_doc_ids": retrieved_ids_ordered,
                            "ground_truth_ids": qa.kb_ids,
                            "collection_name": args.collection_name,
                            "model_name": args.model_name,
                            "top_k": args.top_k,
                        }
                        results_file.write(json.dumps(record) + "\n")
                    progress.update(1)
    finally:
        if results_file:
            results_file.close()

    summary = {
        "num_samples": len(per_question_metrics),
        "exact_match": mean(m["exact_match"] for m in per_question_metrics)
        if per_question_metrics
        else 0.0,
        "containment": mean(m["containment"] for m in per_question_metrics)
        if per_question_metrics
        else 0.0,
        "similarity": mean(m["similarity"] for m in per_question_metrics)
        if per_question_metrics
        else 0.0,
        "retrieval_recall": retrieval_hits / retrieval_total if retrieval_total else 0.0,
    }
    
    # Aggregate @k metrics
    if per_question_retrieval_metrics:
        for k in k_values:
            summary[f"recall@{k}"] = mean(m[f"recall@{k}"] for m in per_question_retrieval_metrics)
            summary[f"precision@{k}"] = mean(m[f"precision@{k}"] for m in per_question_retrieval_metrics)
            summary[f"hit_rate@{k}"] = mean(m[f"hit_rate@{k}"] for m in per_question_retrieval_metrics)
            summary[f"mrr@{k}"] = mean(m[f"mrr@{k}"] for m in per_question_retrieval_metrics)
    else:
        for k in k_values:
            summary[f"recall@{k}"] = 0.0
            summary[f"precision@{k}"] = 0.0
            summary[f"hit_rate@{k}"] = 0.0
            summary[f"mrr@{k}"] = 0.0

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary}
        save_path.write_text(json.dumps(payload, indent=2))

    print("\n=== RAG Evaluation Summary ===")
    print(f"Samples evaluated : {summary['num_samples']}")
    print(f"\n--- Answer Quality Metrics ---")
    print(f"Exact match       : {summary['exact_match']:.3f}")
    print(f"Containment       : {summary['containment']:.3f}")
    print(f"Similarity        : {summary['similarity']:.3f}")
    print(f"Retrieval recall  : {summary['retrieval_recall']:.3f}")
    
    if per_question_retrieval_metrics:
        print(f"\n--- Retrieval Metrics @k ---")
        for k in k_values:
            print(f"Recall@{k}           : {summary[f'recall@{k}']:.3f}")
            print(f"Precision@{k}        : {summary[f'precision@{k}']:.3f}")
            print(f"Hit rate@{k}         : {summary[f'hit_rate@{k}']:.3f}")
            print(f"MRR@{k}              : {summary[f'mrr@{k}']:.3f}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on WixQA.")
    parser.add_argument("--split", type=str, default="train", help="QA split to evaluate.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Randomly sample this many QA pairs (use 0 for all).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="wixqa_recursive_chunks",
        help="Chroma collection to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF model identifier for the generator.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=emb_cfg.model_name,
        help="HF embedding model to use for retrieval.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=emb_cfg.dim,
        help="Embedding dimension (for bookkeeping).",
    )
    parser.add_argument(
        "--reranker-model-name",
        type=str,
        default="BAAI/bge-reranker-base",
        help="HF identifier for the cross-encoder reranker (set empty string to skip).",
    )
    parser.add_argument(
        "--reranker-top-k",
        type=int,
        default=None,
        help="Limit how many reranked docs to pass to the generator (default: use all).",
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Max token length for reranker inputs.",
    )
    parser.add_argument(
        "--reranker-device",
        type=str,
        default=None,
        help="Optional torch device for the reranker (defaults to auto).",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Skip reranking entirely.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="evaluation/rag_answer_eval.json",
        help="Optional path to persist aggregated summary JSON.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="evaluation/rag_results.jsonl",
        help="Path to append per-sample JSONL rows (retrieved/relevant IDs).",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Truncate the results file before writing new rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of QA pairs to process per evaluation batch (sequential batches).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of k values for @k metrics (e.g., --k-values 1 3 5 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rag_answers(args)

