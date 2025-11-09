#!/bin/bash

# Parse command line arguments
CHUNK_STRATEGY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            CHUNK_STRATEGY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --strategy <chunk_strategy>"
            exit 1
            ;;
    esac
done

# Validate strategy parameter
if [ -z "$CHUNK_STRATEGY" ]; then
    echo "Error: --strategy parameter is required"
    echo "Usage: $0 --strategy <recursive|token>"
    exit 1
fi

echo "============================================================"
echo "EVALUATING WITH CHUNK STRATEGY: ${CHUNK_STRATEGY}"
echo "============================================================"

echo ""
echo "============================================================"
echo "MODEL 1/2: Qwen/Qwen3-Embedding-0.6B"
echo "============================================================"

echo "[STEP 1] Building VectorStore for Qwen..."
uv run src/vectorstore/build_vectordb.py \
--chunks data/chunks/chunks_${CHUNK_STRATEGY}_380_50.jsonl \
--embedding_model "Qwen/Qwen3-Embedding-0.6B"

echo "[STEP 2] Evaluating Qwen - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_${CHUNK_STRATEGY}_380_50_qwen_qwen3_embedding_0_6b \
--qdrant_path langchain_qdrant \
--embedding_model "Qwen/Qwen3-Embedding-0.6B" \
--retrieval_mode dense \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "[STEP 3] Evaluating Qwen - Hybrid mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_${CHUNK_STRATEGY}_380_50_qwen_qwen3_embedding_0_6b \
--qdrant_path langchain_qdrant \
--embedding_model "Qwen/Qwen3-Embedding-0.6B" \
--retrieval_mode hybrid \
--chunks_file data/chunks/chunks_${CHUNK_STRATEGY}_380_50.jsonl \
--alpha 0.7 \
--rrf_k 60 \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "✅ Completed Model 1/2: Qwen"
echo ""

echo "============================================================"
echo "MODEL 2/2: BAAI/bge-m3"
echo "============================================================"

echo "[STEP 1] Building VectorStore for BAAI/bge-m3..."
uv run src/vectorstore/build_vectordb.py \
--chunks data/chunks/chunks_${CHUNK_STRATEGY}_380_50.jsonl \
--embedding_model "BAAI/bge-m3"

echo "[STEP 2] Evaluating BAAI/bge-m3 - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_${CHUNK_STRATEGY}_380_50_baai_bge_m3 \
--qdrant_path langchain_qdrant \
--embedding_model "BAAI/bge-m3" \
--retrieval_mode dense \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "[STEP 3] Evaluating BAAI/bge-m3 - Hybrid mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection chunks_${CHUNK_STRATEGY}_380_50_baai_bge_m3 \
--qdrant_path langchain_qdrant \
--embedding_model "BAAI/bge-m3" \
--retrieval_mode hybrid \
--chunks_file data/chunks/chunks_${CHUNK_STRATEGY}_380_50.jsonl \
--alpha 0.7 \
--rrf_k 60 \
--top_k 5 \
--eval_method both \
--retrieve_k 20 \
--agg_mode max \
--cosine_threshold 0.70

echo "✅ Completed Model 2/2: BAAI/bge-m3"
echo ""

echo "============================================================"
echo "🎉 ALL EVALUATIONS COMPLETED!"
echo "============================================================"
echo "Strategy: ${CHUNK_STRATEGY}"
echo "Models evaluated: 2 (Qwen, BAAI/bge-m3)"
echo "Modes: Dense + Hybrid"
echo "Results saved to: data/evaluate_results/evaluation_results.csv"
echo "============================================================"