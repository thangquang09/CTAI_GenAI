#!/bin/bash
# Example commands to run RAG evaluation and compute metrics

# ============================================
# 1. Run eval_rag_answers.py with reranker
# ============================================
# Basic evaluation with reranker (default settings)
python scripts/eval_rag_answers.py \
    --split train \
    --sample-size 50 \
    --collection-name wixqa_recursive_chunks \
    --model-name "Qwen/Qwen3-0.6B" \
    --top-k 5 \
    --max-new-tokens 400 \
    --temperature 0.7 \
    --reranker-model-name "BAAI/bge-reranker-base" \
    --reranker-top-k 3 \
    --results-path evaluation/rag_results.jsonl \
    --save-path evaluation/rag_answer_eval.json \
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
    --top-k 5 \
    --max-new-tokens 400 \
    --temperature 0.7 \
    --disable-reranker \
    --results-path evaluation/rag_results_no_rerank.jsonl \
    --save-path evaluation/rag_answer_eval_no_rerank.json \
    --overwrite-results

# ============================================
# 3. Run compute_rag_metrics.py
# ============================================
# Compute retrieval metrics from the results
python scripts/compute_rag_metrics.py \
    --results-path evaluation/rag_results.jsonl \
    --save-path evaluation/rag_metrics.json

# ============================================
# 4. Full evaluation (all samples, no sampling)
# ============================================
# Use sample-size 0 to evaluate all samples
python scripts/eval_rag_answers.py \
    --split train \
    --sample-size 0 \
    --collection-name wixqa_recursive_chunks \
    --model-name "Qwen/Qwen3-0.6B" \
    --top-k 10 \
    --reranker-model-name "BAAI/bge-reranker-base" \
    --reranker-top-k 5 \
    --results-path evaluation/rag_results_full.jsonl \
    --save-path evaluation/rag_answer_eval_full.json \
    --overwrite-results

# Then compute metrics:
python scripts/compute_rag_metrics.py \
    --results-path evaluation/rag_results_full.jsonl \
    --save-path evaluation/rag_metrics_full.json

