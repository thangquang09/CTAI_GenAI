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


def _per_sample_metrics(record: Dict) -> Optional[Dict[str, float]]:
    retrieved: List[str] = [doc for doc in record.get("retrieved_doc_ids", []) if doc]
    relevant_raw = record.get("ground_truth_ids") or record.get("article_ids") or []
    relevant = [doc for doc in relevant_raw if doc]

    if not relevant:
        return None

    relevant_set = set(relevant)
    hits = [doc for doc in retrieved if doc in relevant_set]

    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    hit = 1.0 if hits else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "recall": recall,
        "precision": precision,
        "hit": hit,
        "mrr": mrr,
    }


def compute_metrics(results_path: Path) -> Dict[str, float]:
    per_sample: List[Dict[str, float]] = []
    total_records = 0
    for record in _load_jsonl(results_path):
        total_records += 1
        metrics = _per_sample_metrics(record)
        if metrics is not None:
            per_sample.append(metrics)

    if not per_sample:
        return {
            "num_records": total_records,
            "num_with_ground_truth": 0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    return {
        "num_records": total_records,
        "num_with_ground_truth": len(per_sample),
        "recall": mean(m["recall"] for m in per_sample),
        "precision": mean(m["precision"] for m in per_sample),
        "hit_rate": mean(m["hit"] for m in per_sample),
        "mrr": mean(m["mrr"] for m in per_sample),
    }


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file '{results_path}' not found. Run eval_rag_answers.py first."
        )

    summary = compute_metrics(results_path)

    print("\n=== Retrieval Metrics ===")
    print(f"Samples (all)       : {summary['num_records']}")
    print(f"Samples (with GT)   : {summary['num_with_ground_truth']}")
    print(f"Recall              : {summary['recall']:.3f}")
    print(f"Precision           : {summary['precision']:.3f}")
    print(f"Hit rate            : {summary['hit_rate']:.3f}")
    print(f"MRR                 : {summary['mrr']:.3f}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

