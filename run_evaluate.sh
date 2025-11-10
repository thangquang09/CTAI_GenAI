#!/bin/bash

# run_evaluate_v2.sh
# Advanced evaluation script with:
# - Mock mode for fast testing
# - Experiments config from JSONL file
# - Smart skip logic (skip if collection/chunks exist)

set -e  # Exit on error

# Suppress Python warnings (including Qdrant local mode warning)
export PYTHONWARNINGS="ignore"

# Default values
EXPERIMENTS_FILE="experiments.jsonl"
MOCK=0
QDRANT_PATH="langchain_qdrant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments)
            EXPERIMENTS_FILE="$2"
            shift 2
            ;;
        --mock)
            MOCK=1
            shift
            ;;
        --qdrant_path)
            QDRANT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiments FILE    Path to experiments JSONL file (default: experiments.jsonl)"
            echo "  --mock               Enable mock mode (use limited data for fast testing)"
            echo "  --qdrant_path PATH   Path to Qdrant storage (default: langchain_qdrant)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --experiments experiments.jsonl"
            echo "  $0 --experiments experiments.jsonl --mock"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate experiments file exists
if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo -e "${RED}Error: Experiments file not found: ${EXPERIMENTS_FILE}${NC}"
    exit 1
fi

# Print configuration
echo "============================================================"
echo "EVALUATION PIPELINE V2 - CONFIGURATION"
echo "============================================================"
echo "Experiments file: ${EXPERIMENTS_FILE}"
echo "Mock mode: $([ $MOCK -eq 1 ] && echo 'ENABLED (fast testing)' || echo 'DISABLED (full run)')"
echo "Qdrant path: ${QDRANT_PATH}"
echo "============================================================"
echo ""

# Python helper script to parse experiments and generate collection names
read -r -d '' PYTHON_HELPER << 'EOF' || true
import json
import sys
import re
import os

def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name - EXACT COPY from build_vectordb.py"""
    name = name.lower()
    name = re.sub(r'[/\-.]', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > 255:
        name = name[:255]
    return name

def get_collection_name(chunks_file: str, embedding_model: str, mode: str = "dense") -> str:
    """Generate collection name from config"""
    chunks_basename = os.path.basename(chunks_file).replace(".jsonl", "")
    embedding_safe = sanitize_collection_name(embedding_model)
    collection_name = sanitize_collection_name(f"{chunks_basename}_{embedding_safe}")
    
    # Add mode suffix
    if mode == "native_hybrid":
        collection_name = f"{collection_name}_native_hybrid"
    elif mode == "hybrid":
        collection_name = f"{collection_name}_hybrid"
    
    return collection_name

if __name__ == "__main__":
    command = sys.argv[1]
    
    if command == "parse":
        # Parse experiments JSONL file
        experiments_file = sys.argv[2]
        with open(experiments_file, 'r') as f:
            experiments = json.load(f)
        
        # Output as JSON for bash to parse
        print(json.dumps(experiments))
    
    elif command == "collection_name":
        # Generate collection name
        chunks_file = sys.argv[2]
        embedding_model = sys.argv[3]
        mode = sys.argv[4] if len(sys.argv) > 4 else "dense"
        
        collection_name = get_collection_name(chunks_file, embedding_model, mode)
        print(collection_name)
EOF

# Function to check if collection exists
collection_exists() {
    local collection_name=$1
    local qdrant_path=$2
    
    # Check if collection directory exists in Qdrant storage
    if [ -d "${qdrant_path}/collection/${collection_name}" ]; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

# Function to check if chunks file exists
chunks_exists() {
    local chunks_file=$1
    
    if [ -f "$chunks_file" ]; then
        return 0  # exists
    else
        return 1  # not exists
    fi
}

# Parse experiments using Python helper
echo -e "${BLUE}[INFO] Parsing experiments from ${EXPERIMENTS_FILE}...${NC}"
EXPERIMENTS_JSON=$(python -c "$PYTHON_HELPER" parse "$EXPERIMENTS_FILE")

# Get number of experiments
NUM_EXPERIMENTS=$(echo "$EXPERIMENTS_JSON" | python -c "import sys, json; print(len(json.load(sys.stdin)))")
echo -e "${GREEN}[INFO] Found ${NUM_EXPERIMENTS} experiment(s) to run${NC}"
echo ""

# Loop through experiments
for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
    echo "============================================================"
    echo "EXPERIMENT $((i + 1))/${NUM_EXPERIMENTS}"
    echo "============================================================"
    
    # Extract experiment config
    EXP_JSON=$(echo "$EXPERIMENTS_JSON" | python -c "import sys, json; print(json.dumps(json.load(sys.stdin)[$i]))")
    
    STRATEGY=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('strategy', 'recursive'))")
    CHUNK_SIZE=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('chunk_size', 380))")
    OVERLAP=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('overlap', 50))")
    EMBEDDING_MODEL=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('embedding_model', 'BAAI/bge-small-en-v1.5'))")
    MODE=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('mode', 'dense'))")
    
    # Optional params
    TOP_K=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('top_k', 5))")
    RETRIEVE_K=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('retrieve_k', 20))")
    AGG_MODE=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('agg_mode', 'max'))")
    COSINE_THRESHOLD=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('cosine_threshold', 0.70))")
    EVAL_METHOD=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('eval_method', 'both'))")
    ALPHA=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('alpha', 0.7))")
    RRF_K=$(echo "$EXP_JSON" | python -c "import sys, json; print(json.load(sys.stdin).get('rrf_k', 60))")
    
    # Determine chunks file and collection name
    CHUNKS_FILE="data/chunks/chunks_${STRATEGY}_${CHUNK_SIZE}_${OVERLAP}.jsonl"
    COLLECTION_NAME=$(python -c "$PYTHON_HELPER" collection_name "$CHUNKS_FILE" "$EMBEDDING_MODEL" "$MODE")
    
    # Print experiment config
    echo "Configuration:"
    echo "  Strategy: ${STRATEGY}"
    echo "  Chunk size: ${CHUNK_SIZE}"
    echo "  Overlap: ${OVERLAP}"
    echo "  Embedding model: ${EMBEDDING_MODEL}"
    echo "  Mode: ${MODE}"
    echo "  Chunks file: ${CHUNKS_FILE}"
    echo "  Collection: ${COLLECTION_NAME}"
    echo ""
    
    # Mock mode settings
    if [ $MOCK -eq 1 ]; then
        echo -e "${YELLOW}[MOCK MODE] Using limited data for fast testing${NC}"
        LIMIT_DOCS=100
        MAX_QUERIES=10
        RECREATE_FLAG="--recreate"
    else
        LIMIT_DOCS=0
        MAX_QUERIES=0
        RECREATE_FLAG=""
    fi
    
    # ========================================
    # STEP 1: CHUNKING
    # ========================================
    SKIP_CHUNKING=0
    
    if [ $MOCK -eq 0 ] && chunks_exists "$CHUNKS_FILE"; then
        echo -e "${GREEN}[SKIP] Chunks file already exists: ${CHUNKS_FILE}${NC}"
        SKIP_CHUNKING=1
    else
        echo -e "${BLUE}[STEP 1] Chunking data...${NC}"
        
        # Chunking always runs on full data (no limit even in mock mode)
        CMD="uv run src/chunking/chunk_wixqa.py \
--strategy ${STRATEGY} \
--chunk_size ${CHUNK_SIZE} \
--overlap ${OVERLAP}"
        
        echo "Command: ${CMD}"
        eval $CMD
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[SUCCESS] Chunking completed${NC}"
        else
            echo -e "${RED}[ERROR] Chunking failed${NC}"
            continue
        fi
    fi
    echo ""
    
    # ========================================
    # STEP 2: BUILD VECTOR STORE
    # ========================================
    SKIP_INDEXING=0
    
    if [ $MOCK -eq 0 ] && collection_exists "$COLLECTION_NAME" "$QDRANT_PATH"; then
        echo -e "${GREEN}[SKIP] Collection already exists: ${COLLECTION_NAME}${NC}"
        SKIP_INDEXING=1
    else
        echo -e "${BLUE}[STEP 2] Building vector store...${NC}"
        
        CMD="uv run src/vectorstore/build_vectordb.py \
--chunks ${CHUNKS_FILE} \
--embedding_model \"${EMBEDDING_MODEL}\" \
--mode ${MODE} \
--out_dir ${QDRANT_PATH}"
        
        if [ $MOCK -eq 1 ]; then
            CMD="${CMD} --limit ${LIMIT_DOCS} ${RECREATE_FLAG}"
            echo -e "${YELLOW}[MOCK] Limiting to ${LIMIT_DOCS} documents and recreating collection${NC}"
        fi
        
        echo "Command: ${CMD}"
        eval $CMD
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[SUCCESS] Vector store built${NC}"
        else
            echo -e "${RED}[ERROR] Vector store build failed${NC}"
            continue
        fi
    fi
    echo ""
    
    # ========================================
    # STEP 3: EVALUATE
    # ========================================
    echo -e "${BLUE}[STEP 3] Evaluating retrieval...${NC}"
    
    CMD="uv run src/evaluate.py \
--mode from_collection \
--collection ${COLLECTION_NAME} \
--qdrant_path ${QDRANT_PATH} \
--embedding_model \"${EMBEDDING_MODEL}\" \
--retrieval_mode ${MODE} \
--top_k ${TOP_K} \
--eval_method ${EVAL_METHOD} \
--retrieve_k ${RETRIEVE_K} \
--agg_mode ${AGG_MODE} \
--cosine_threshold ${COSINE_THRESHOLD}"
    
    # Add mode-specific params
    if [ "$MODE" = "hybrid" ]; then
        CMD="${CMD} --chunks_file ${CHUNKS_FILE} --alpha ${ALPHA} --rrf_k ${RRF_K}"
    fi
    
    # Add mock mode limit
    if [ $MOCK -eq 1 ]; then
        CMD="${CMD} --max_queries ${MAX_QUERIES}"
        echo -e "${YELLOW}[MOCK] Limiting to ${MAX_QUERIES} queries${NC}"
    fi
    
    echo "Command: ${CMD}"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Evaluation completed${NC}"
    else
        echo -e "${RED}[ERROR] Evaluation failed${NC}"
        continue
    fi
    echo ""
    
    # Summary for this experiment
    echo -e "${GREEN}✅ Experiment $((i + 1))/${NUM_EXPERIMENTS} completed${NC}"
    echo "  Collection: ${COLLECTION_NAME}"
    echo "  Chunking: $([ $SKIP_CHUNKING -eq 1 ] && echo 'SKIPPED' || echo 'EXECUTED')"
    echo "  Indexing: $([ $SKIP_INDEXING -eq 1 ] && echo 'SKIPPED' || echo 'EXECUTED')"
    echo "  Evaluation: EXECUTED"
    echo "============================================================"
    echo ""
done

# Final summary
echo "============================================================"
echo "🎉 ALL EXPERIMENTS COMPLETED!"
echo "============================================================"
echo "Total experiments: ${NUM_EXPERIMENTS}"
echo "Mock mode: $([ $MOCK -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo ""
echo "Results saved to: data/evaluate_results/evaluation_results.csv"
echo "============================================================"
