import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _per_sample_metrics(record: Dict, k_values: List[int]) -> Optional[Dict[str, float]]:
    retrieved: List[str] = [doc for doc in record.get("retrieved_doc_ids", []) if doc]
    relevant_raw = record.get("ground_truth_ids") or record.get("article_ids") or []
    relevant = [doc for doc in relevant_raw if doc]

    if not relevant:
        return None

    relevant_set = set(relevant)
    
    # Calculate metrics for each k value
    metrics: Dict[str, float] = {}
    
    for k in k_values:
        retrieved_at_k = retrieved[:k]
        hits_at_k = [doc for doc in retrieved_at_k if doc in relevant_set]
        
        recall_at_k = len(hits_at_k) / len(relevant) if relevant else 0.0
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
    
    # Also compute overall metrics (using all retrieved)
    hits = [doc for doc in retrieved if doc in relevant_set]
    metrics["recall"] = len(hits) / len(relevant) if relevant else 0.0
    metrics["precision"] = len(hits) / len(retrieved) if retrieved else 0.0
    metrics["hit"] = 1.0 if hits else 0.0
    
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr
    
    return metrics


def compute_metrics(results_path: Path, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    per_sample: List[Dict[str, float]] = []
    total_records = 0
    for record in _load_jsonl(results_path):
        total_records += 1
        metrics = _per_sample_metrics(record, k_values)
        if metrics is not None:
            per_sample.append(metrics)

    if not per_sample:
        result = {
            "num_records": total_records,
            "num_with_ground_truth": 0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }
        for k in k_values:
            result[f"recall@{k}"] = 0.0
            result[f"precision@{k}"] = 0.0
            result[f"hit_rate@{k}"] = 0.0
            result[f"mrr@{k}"] = 0.0
        return result

    result = {
        "num_records": total_records,
        "num_with_ground_truth": len(per_sample),
        "recall": mean(m["recall"] for m in per_sample),
        "precision": mean(m["precision"] for m in per_sample),
        "hit_rate": mean(m["hit"] for m in per_sample),
        "mrr": mean(m["mrr"] for m in per_sample),
    }
    
    # Aggregate @k metrics
    for k in k_values:
        result[f"recall@{k}"] = mean(m[f"recall@{k}"] for m in per_sample)
        result[f"precision@{k}"] = mean(m[f"precision@{k}"] for m in per_sample)
        result[f"hit_rate@{k}"] = mean(m[f"hit_rate@{k}"] for m in per_sample)
        result[f"mrr@{k}"] = mean(m[f"mrr@{k}"] for m in per_sample)
    
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute retrieval metrics from rag_results.jsonl."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="evaluation/rag_results.jsonl",
        help="Path to the JSONL file produced by eval_rag_answers.py",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="evaluation/rag_metrics.json",
        help="Optional path to write the aggregated metrics JSON.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of k values for @k metrics (e.g., --k-values 1 3 5 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file '{results_path}' not found. Run eval_rag_answers.py first."
        )

    summary = compute_metrics(results_path, k_values=args.k_values)

    print("\n=== Retrieval Metrics ===")
    print(f"Samples (all)       : {summary['num_records']}")
    print(f"Samples (with GT)   : {summary['num_with_ground_truth']}")
    print(f"\n--- Overall Metrics ---")
    print(f"Recall              : {summary['recall']:.3f}")
    print(f"Precision           : {summary['precision']:.3f}")
    print(f"Hit rate            : {summary['hit_rate']:.3f}")
    print(f"MRR                 : {summary['mrr']:.3f}")
    
    print(f"\n--- Metrics @k ---")
    for k in args.k_values:
        print(f"Recall@{k}           : {summary[f'recall@{k}']:.3f}")
        print(f"Precision@{k}        : {summary[f'precision@{k}']:.3f}")
        print(f"Hit rate@{k}         : {summary[f'hit_rate@{k}']:.3f}")
        print(f"MRR@{k}              : {summary[f'mrr@{k}']:.3f}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

