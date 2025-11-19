from rag_wixqa.data.load_wixqa import load_qa_pairs
from rag_wixqa.chunking.strategies import ChunkMethod
from rag_wixqa.eval.retrieval_eval import evaluate_all_methods


def main():
    print("Loading QA pairs...")
    qa_pairs = load_qa_pairs(split="train")  # or "validation" depending on config

    methods: list[ChunkMethod] = ["recursive", "token", "semantic"]

    print("Evaluating retrieval for all chunking methods...")
    results = evaluate_all_methods(qa_pairs, methods=methods, k=5)

    print("\n=== Retrieval results (Recall@5) ===")
    for m, metrics in results.items():
        print(
            f"{m:10s} | recall@{metrics.k} = {metrics.recall_at_k:.3f} "
            f"(on {metrics.num_queries} queries)"
        )


if __name__ == "__main__":
    main()
