#!/bin/bash

# Default values
CHUNK_STRATEGY=""
CHUNK_SIZE=380
OVERLAP=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            CHUNK_STRATEGY="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --strategy <chunk_strategy> [--chunk_size <size>] [--overlap <overlap>]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$CHUNK_STRATEGY" ]; then
    echo "Error: --strategy parameter is required"
    echo "Usage: $0 --strategy <recursive|token> [--chunk_size <size>] [--overlap <overlap>]"
    exit 1
fi

# Build chunks filename
CHUNKS_FILE="data/chunks/chunks_${CHUNK_STRATEGY}_${CHUNK_SIZE}_${OVERLAP}.jsonl"

# Helper function to get collection name from Python
get_collection_name() {
    local embedding_model=$1
    python -c "from run_evaluate import get_collection_name; print(get_collection_name('${CHUNKS_FILE}', '${embedding_model}'))"
}

echo "============================================================"
echo "EVALUATION CONFIGURATION"
echo "============================================================"
echo "Chunk Strategy: ${CHUNK_STRATEGY}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Overlap: ${OVERLAP}"
echo "Chunks File: ${CHUNKS_FILE}"
echo "============================================================"

echo ""
echo "============================================================"
echo "MODEL 1/2: Qwen/Qwen3-Embedding-0.6B"
echo "============================================================"

# Get collection name dynamically
QWEN_COLLECTION=$(get_collection_name "Qwen/Qwen3-Embedding-0.6B")
echo "Collection name: ${QWEN_COLLECTION}"
echo ""

echo "[STEP 1] Building VectorStore for Qwen..."
uv run src/vectorstore/build_vectordb.py \
--chunks ${CHUNKS_FILE} \
--embedding_model "Qwen/Qwen3-Embedding-0.6B"

echo "[STEP 2] Evaluating Qwen - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection ${QWEN_COLLECTION} \
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
--collection ${QWEN_COLLECTION} \
--qdrant_path langchain_qdrant \
--embedding_model "Qwen/Qwen3-Embedding-0.6B" \
--retrieval_mode hybrid \
--chunks_file ${CHUNKS_FILE} \
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

# Get collection name dynamically
BGE_COLLECTION=$(get_collection_name "BAAI/bge-m3")
echo "Collection name: ${BGE_COLLECTION}"
echo ""

echo "[STEP 1] Building VectorStore for BAAI/bge-m3..."
uv run src/vectorstore/build_vectordb.py \
--chunks ${CHUNKS_FILE} \
--embedding_model "BAAI/bge-m3"

echo "[STEP 2] Evaluating BAAI/bge-m3 - Dense mode..."
uv run src/evaluate.py \
--mode from_collection \
--collection ${BGE_COLLECTION} \
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
--collection ${BGE_COLLECTION} \
--qdrant_path langchain_qdrant \
--embedding_model "BAAI/bge-m3" \
--retrieval_mode hybrid \
--chunks_file ${CHUNKS_FILE} \
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
echo "Configuration:"
echo "  Strategy: ${CHUNK_STRATEGY}"
echo "  Chunk Size: ${CHUNK_SIZE}"
echo "  Overlap: ${OVERLAP}"
echo "  Chunks File: ${CHUNKS_FILE}"
echo ""
echo "Models evaluated: 2 (Qwen, BAAI/bge-m3)"
echo "Modes: Dense + Hybrid"
echo ""
echo "Results saved to: data/evaluate_results/evaluation_results.csv"
echo "============================================================"