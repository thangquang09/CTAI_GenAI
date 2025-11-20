from rag_wixqa.rag.pipeline import Qwen3RAGPipeline, RAGConfig


def main():
    cfg = RAGConfig(
        collection_name="wixqa_recursive_chunks",  # or token/semantic versions
        model_name="Qwen/Qwen3-0.6B",
        top_k=5,
        max_new_tokens=400,
        temperature=0.7,
        enable_thinking=False,  # keep simple
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        use_reranker=True,
        reranker_model_name="BAAI/bge-reranker-base",
        reranker_top_k=3,
    )

    rag = Qwen3RAGPipeline(cfg)

    question = "I need to know if more information is required for Wix payment verification status for my individual account."
    result = rag.answer(question, return_docs=True)

    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== USED CONTEXT DOC IDS ===")
    for d in result["docs"]:
        doc_id = d.metadata.get("doc_id")
        score = d.metadata.get("rerank_score")
        if score is not None:
            print(f"{doc_id} (rerank={score:.4f})")
        else:
            print(doc_id)


if __name__ == "__main__":
    main()
