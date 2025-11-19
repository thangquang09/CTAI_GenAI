import argparse
import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from tqdm import tqdm

from rag_wixqa.data.load_wixqa import load_qa_pairs
from rag_wixqa.data.models import QAPair
from rag_wixqa.rag.pipeline import Qwen3RAGPipeline, RAGConfig


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


def evaluate_rag_answers(args: argparse.Namespace) -> Dict[str, float]:
    qa_pairs = load_qa_pairs(split=args.split)
    qa_pairs = sample_qa_pairs(qa_pairs, args.sample_size)

    rag_cfg = RAGConfig(
        collection_name=args.collection_name,
        model_name=args.model_name,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    rag = Qwen3RAGPipeline(rag_cfg)

    per_question_metrics: List[Dict[str, float]] = []
    retrieval_hits = 0
    retrieval_total = 0

    results_file = None
    if args.results_path:
        results_path = Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite_results else "a"
        results_file = results_path.open(mode, encoding="utf-8")

    try:
        for qa in tqdm(qa_pairs, desc="Evaluating RAG answers"):
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
                    "retrieved_doc_ids": retrieved_ids_ordered,
                    "ground_truth_ids": qa.kb_ids,
                    "collection_name": args.collection_name,
                    "model_name": args.model_name,
                    "top_k": args.top_k,
                }
                results_file.write(json.dumps(record) + "\n")
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

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary}
        save_path.write_text(json.dumps(payload, indent=2))

    print("\n=== RAG Evaluation Summary ===")
    print(f"Samples evaluated : {summary['num_samples']}")
    print(f"Exact match       : {summary['exact_match']:.3f}")
    print(f"Containment       : {summary['containment']:.3f}")
    print(f"Similarity        : {summary['similarity']:.3f}")
    print(f"Retrieval recall  : {summary['retrieval_recall']:.3f}")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rag_answers(args)

