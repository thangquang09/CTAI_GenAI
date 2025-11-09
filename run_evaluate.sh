#!/bin/bash

echo "============================================================"
echo "Build VectorStore for 2 models (chunks_token)"
echo "============================================================"

# uv run src/vectorstore/build_vectordb.py \
# --chunks data/chunks/chunks_token_380_50.jsonl \
# --embedding_model "Qwen/Qwen3-Embedding-0.6B"

echo "Evaluating Qwen - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_token_380_50_qwen_qwen3_embedding_0_6b \
--qdrant_path langchain_qdrant \
--embedding_model "Qwen/Qwen3-Embedding-0.6B" \
--retrieval_mode dense \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "Evaluating Qwen - Hybrid mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_token_380_50_qwen_qwen3_embedding_0_6b \
--qdrant_path langchain_qdrant \
--embedding_model "Qwen/Qwen3-Embedding-0.6B" \
--retrieval_mode hybrid \
--chunks_file data/chunks/chunks_token_380_50.jsonl \
--alpha 0.7 \
--rrf_k 60 \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

uv run src/vectorstore/build_vectordb.py \
--chunks data/chunks/chunks_token_380_50.jsonl \
--embedding_model "BAAI/bge-m3"

echo "Evaluating BAAI/bge-m3 - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_token_380_50_baai_bge_m3 \
--qdrant_path langchain_qdrant \
--embedding_model "BAAI/bge-m3" \
--retrieval_mode dense \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "Evaluating BAAI/bge-m3 - Hybrid mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_token_380_50_baai_bge_m3 \
--qdrant_path langchain_qdrant \
--embedding_model "BAAI/bge-m3" \
--retrieval_mode hybrid \
--chunks_file data/chunks/chunks_token_380_50.jsonl \
--alpha 0.7 \
--rrf_k 60 \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

# echo "============================================================"
# echo "Evaluate Model 1: Qwen/Qwen3-Embedding-0.6B"
# echo "============================================================"


# echo "============================================================"
# echo "Evaluate Model 2: BAAI/bge-m3"
# echo "============================================================"



echo "============================================================"
echo "All evaluations completed!"
echo "Results saved to: data/evaluate_results/evaluation_results.csv"
echo "============================================================"