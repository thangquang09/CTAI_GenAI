"""
Automated evaluation pipeline for multiple embedding models.
Ensures collection names are generated consistently with build_vectordb.py logic.
"""
import os
import re
import subprocess
import sys
from typing import List


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name - EXACT COPY from build_vectordb.py
    """
    name = name.lower()
    name = re.sub(r'[/\-.]', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > 255:
        name = name[:255]
    return name


def get_collection_name(chunks_file: str, embedding_model: str) -> str:
    """
    Generate collection name - EXACT LOGIC from build_vectordb.py
    """
    chunks_basename = os.path.basename(chunks_file)
    base_name = chunks_basename.replace(".jsonl", "")
    embedding_safe = sanitize_collection_name(embedding_model)
    collection_name = f"{base_name}_{embedding_safe}"
    return sanitize_collection_name(collection_name)


def run_command(cmd: List[str], description: str) -> bool:
    """
    Execute a command and return success status.
    """
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    # Configuration
    CHUNKS_FILE = "data/chunks/chunks_recursive_380_50.jsonl"
    QDRANT_PATH = "langchain_qdrant"
    
    # Embedding models to evaluate
    EMBEDDING_MODELS = [
        "Qwen/Qwen3-Embedding-0.6B",
        "google/embeddinggemma-300m",
        "BAAI/bge-m3",
    ]
    
    # Evaluation configs
    EVAL_CONFIG = {
        "top_k": 5,
        "max_queries": 50,
        "retrieve_k": 50,
        "agg_mode": "max",
        "cosine_threshold": 0.75,
        "eval_method": "both",
        "alpha": 0.7,  # For hybrid mode
        "rrf_k": 60,   # For hybrid mode
    }
    
    print("="*80)
    print("AUTOMATED EVALUATION PIPELINE")
    print("="*80)
    print(f"Chunks file: {CHUNKS_FILE}")
    print(f"Qdrant path: {QDRANT_PATH}")
    print(f"Models to evaluate: {len(EMBEDDING_MODELS)}")
    for model in EMBEDDING_MODELS:
        collection = get_collection_name(CHUNKS_FILE, model)
        print(f"  - {model}")
        print(f"    → Collection: {collection}")
    print("="*80)
    
    # Step 1: Chunking
    print("\n" + "="*80)
    print("STEP 1: CHUNKING DATA")
    print("="*80)
    
    chunking_commands = [
        (
            ["uv", "run", "src/chunking/chunk_wixqa.py",
             "--strategy", "recursive",
             "--chunk_size", "380",
             "--overlap", "50"],
            "Chunking with recursive strategy"
        ),
        (
            ["uv", "run", "src/chunking/chunk_wixqa.py",
             "--strategy", "token",
             "--chunk_size", "380",
             "--overlap", "50"],
            "Chunking with token strategy"
        ),
    ]
    
    for cmd, desc in chunking_commands:
        if not run_command(cmd, desc):
            print("⚠️ Chunking failed, but continuing...")
    
    # Step 2 & 3: Build VectorStore -> Evaluate (per model)
    print("\n" + "="*80)
    print("STEP 2 & 3: BUILD VECTORSTORE -> EVALUATE (PER MODEL)")
    print("="*80)
    print("Strategy: Build one model, evaluate immediately, then move to next")
    print("Benefits: Lower memory usage, easier error recovery")
    print("Note: --recreate is commented out to avoid rebuilding existing collections")
    print("="*80)
    
    results_summary = []
    
    for i, model in enumerate(EMBEDDING_MODELS, 1):
        collection = get_collection_name(CHUNKS_FILE, model)
        
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(EMBEDDING_MODELS)}: {model}")
        print(f"Collection: {collection}")
        print(f"{'='*80}")
        
        # Step 2a: Build VectorStore for this model
        print(f"\n[STEP 2a] Building VectorStore for {model}...")
        build_cmd = [
            "uv", "run", "src/vectorstore/build_vectordb.py",
            "--chunks", CHUNKS_FILE,
            "--embedding_model", model,
            "--limit", "0",
            # "--recreate",  # Uncomment to rebuild
        ]
        
        build_success = run_command(build_cmd, f"Building collection: {collection}")
        
        if not build_success:
            print(f"❌ Failed to build collection for {model}")
            print("   Skipping evaluation for this model...")
            results_summary.append({
                "model": model,
                "collection": collection,
                "build": "❌",
                "dense": "⏭️",
                "hybrid": "⏭️",
            })
            continue
        
        print(f"✅ Collection built successfully: {collection}")
        
        # Step 3a: Evaluate Dense mode
        print(f"\n[STEP 3a] Evaluating {model} - Dense mode...")
        dense_cmd = [
            "uv", "run", "src/evaluate.py",
            "--mode", "from_collection",
            "--collection", collection,
            "--embedding_model", model,
            "--retrieval_mode", "dense",
            "--top_k", str(EVAL_CONFIG["top_k"]),
            "--eval_method", EVAL_CONFIG["eval_method"],
            "--max_queries", str(EVAL_CONFIG["max_queries"]),
            "--retrieve_k", str(EVAL_CONFIG["retrieve_k"]),
            "--agg_mode", EVAL_CONFIG["agg_mode"],
            "--cosine_threshold", str(EVAL_CONFIG["cosine_threshold"]),
        ]
        
        dense_success = run_command(dense_cmd, f"{model} - Dense mode")
        
        # Step 3b: Evaluate Hybrid mode
        print(f"\n[STEP 3b] Evaluating {model} - Hybrid mode...")
        hybrid_cmd = [
            "uv", "run", "src/evaluate.py",
            "--mode", "from_collection",
            "--collection", collection,
            "--embedding_model", model,
            "--retrieval_mode", "hybrid",
            "--chunks_file", CHUNKS_FILE,
            "--alpha", str(EVAL_CONFIG["alpha"]),
            "--rrf_k", str(EVAL_CONFIG["rrf_k"]),
            "--top_k", str(EVAL_CONFIG["top_k"]),
            "--eval_method", EVAL_CONFIG["eval_method"],
            "--max_queries", str(EVAL_CONFIG["max_queries"]),
            "--retrieve_k", str(EVAL_CONFIG["retrieve_k"]),
            "--agg_mode", EVAL_CONFIG["agg_mode"],
            "--cosine_threshold", str(EVAL_CONFIG["cosine_threshold"]),
        ]
        
        hybrid_success = run_command(hybrid_cmd, f"{model} - Hybrid mode")
        
        # Save results for this model
        results_summary.append({
            "model": model,
            "collection": collection,
            "build": "✅",
            "dense": "✅" if dense_success else "❌",
            "hybrid": "✅" if hybrid_success else "❌",
        })
        
        print(f"\n{'='*80}")
        print(f"COMPLETED MODEL {i}/{len(EMBEDDING_MODELS)}: {model}")
        print(f"Build: ✅ | Dense: {'✅' if dense_success else '❌'} | Hybrid: {'✅' if hybrid_success else '❌'}")
        print(f"{'='*80}")
    
    # Final Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETED - SUMMARY")
    print("="*80)
    print(f"\n{'Model':<40} {'Build':<10} {'Dense':<10} {'Hybrid':<10}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['model']:<40} {result['build']:<10} {result['dense']:<10} {result['hybrid']:<10}")
    
    print("\n" + "="*80)
    print("Results saved to: data/evaluate_results/evaluation_results.csv")
    print("="*80)
    
    # Check if any evaluations failed
    failed = [r for r in results_summary if r['build'] == "❌" or r['dense'] == "❌" or r['hybrid'] == "❌"]
    if failed:
        print(f"\n⚠️ WARNING: {len(failed)} model(s) had evaluation failures")
        return 1
    
    print("\n✅ All evaluations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
