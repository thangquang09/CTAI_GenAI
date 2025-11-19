#!/bin/bash
# Example commands to run RAG evaluation and compute metrics


python scripts/ingest_kb.py \
    --method recursive \
    --embedding-model-name "Qwen/Qwen3-Embedding-0.6B" \
    --embedding-dim 1024 \
    --chunk-batch-size 128 \
    --chroma-batch-size 4096

# --method recursive \ or --method semantic or --method token

# ============================================
# 1. Run eval_rag_answers.py with reranker
# ============================================
# Basic evaluation with reranker (default settings)
python scripts/eval_rag_answers.py \
    --split train \
    --sample-size 200 \
    --collection-name wixqa_recursive_chunks \
    --model-name "Qwen/Qwen3-0.6B" \
    --embedding-model-name "Qwen/Qwen3-Embedding-0.6B" \
    --embedding-dim 1024 \
    --batch-size 32 \
    --top-k 5 \
    --max-new-tokens 400 \
    --temperature 0.7 \
    --reranker-model-name "BAAI/bge-reranker-base" \
    --reranker-top-k 3 \
    --k-values 1 3 5 10 \
    --results-path evaluation/rag_results_200s_Qwen306B_embedalso_topk5_400max_07temp_reranker3.jsonl \
    --save-path evaluation/rag_answer_eval_200s_Qwen306B_embedalso_topk5_400max_07temp_reranker3.json \
    --overwrite-results

# ============================================
# 2. Run eval_rag_answers.py without reranker
# ============================================
# For comparison: evaluation without reranker
python scripts/eval_rag_answers.py \
    --split train \
    --sample-size 50 \
    --collection-name wixqa_recursive_chunks \
    --model-name "Qwen/Qwen3-0.6B" \
    --embedding-model-name "Qwen/Qwen3-Embedding-0.6B" \
    --embedding-dim 1024 \
    --batch-size 32 \
    --top-k 5 \
    --max-new-tokens 400 \
    --temperature 0.7 \
    --disable-reranker \
    --k-values 1 3 5 10 \
    --results-path evaluation/rag_results_no_rerank.jsonl \
    --save-path evaluation/rag_answer_eval_no_rerank.json \
    --overwrite-results

# ============================================
# 3. Run compute_rag_metrics.py
# ============================================
# Compute retrieval metrics from the results
python scripts/compute_rag_metrics.py \
    --results-path evaluation/rag_results_200s_Qwen306B_embedalso_topk5_400max_07temp_reranker3.jsonl \
    --save-path evaluation/rag_metrics.json \
    --k-values 1 3 5 10

# ============================================
# 4. Full evaluation (all samples, no sampling)
# ============================================
# Use sample-size 0 to evaluate all samples
python scripts/eval_rag_answers.py \
    --split train \
    --sample-size 0 \
    --collection-name wixqa_recursive_chunks \
    --model-name "Qwen/Qwen3-0.6B" \
    --embedding-model-name "Qwen/Qwen3-Embedding-0.6B" \
    --embedding-dim 1024 \
    --batch-size 64 \
    --top-k 10 \
    --reranker-model-name "BAAI/bge-reranker-base" \
    --reranker-top-k 5 \
    --k-values 1 3 5 10 \
    --results-path evaluation/rag_results_full.jsonl \
    --save-path evaluation/rag_answer_eval_full.json \
    --overwrite-results

# Then compute metrics:
python scripts/compute_rag_metrics.py \
    --results-path evaluation/rag_results_full.jsonl \
    --save-path evaluation/rag_metrics_full.json \
    --k-values 1 3 5 10 20

