"""
Evaluation script for RAG retrieval system.

Đánh giá chất lượng retrieval với các metrics:
- Hit Rate (Recall@K)
- Mean Reciprocal Rank (MRR)
- Precision@K
- Recall@K

Pipeline modes:
- full: chunk → vectorstore → retriever → evaluate (pipeline hoàn chỉnh)
- from_collection: load collection → retriever → evaluate (đã có vectorstore)
"""

# Standard library imports
import argparse
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

# Third-party imports
import numpy as np
from datasets import load_dataset
from langchain_core.retrievers import BaseRetriever
from tqdm import tqdm

# Local imports
from vectorstore.build_retriever import build_retriever


# ==================== PIPELINE FUNCTIONS ====================

def run_chunking_pipeline(
    strategy: str = "recursive",
    chunk_size: int = 380,
    overlap: int = 50,
    max_records: int = 0,
    output_dir: str = "data/chunks",
) -> str:
    """
    Chạy chunking pipeline để tạo chunks từ WixQA dataset.
    
    Args:
        strategy: Chiến lược chunking (recursive, token, etc.)
        chunk_size: Kích thước chunk
        overlap: Số token/char overlap
        max_records: Giới hạn số bài (0 = toàn bộ)
        output_dir: Thư mục output
    
    Returns:
        str: Đường dẫn file chunks đã tạo
    """
    print("\n" + "="*80)
    print("STEP 1: CHUNKING")
    print("="*80)
    
    # Tạo output directory nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Tên file output
    chunks_file = os.path.join(
        output_dir, 
        f"chunks_{strategy}_{chunk_size}_{overlap}.jsonl"
    )
    
    # Build command
    cmd = [
        sys.executable,  # python executable
        "src/chunking/chunk_wixqa.py",
        "--strategy", strategy,
        "--chunk_size", str(chunk_size),
        "--overlap", str(overlap),
        "--out", chunks_file,
    ]
    
    if max_records > 0:
        cmd.extend(["--max_records", str(max_records)])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Chạy chunking script
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not created: {chunks_file}")
    
    print(f"[SUCCESS] Chunks created: {chunks_file}")
    return chunks_file


def run_vectorstore_pipeline(
    chunks_file: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    collection_name: str = "",
    recreate: bool = False,
    limit: int = 0,
    batch_size: int = 256,
) -> str:
    """
    Chạy vectorstore pipeline để build Qdrant collection từ chunks.
    
    Args:
        chunks_file: Đường dẫn file JSONL chunks
        qdrant_path: Thư mục lưu Qdrant database
        embedding_model: HuggingFace embedding model
        collection_name: Tên collection (auto nếu rỗng)
        recreate: Xóa và tạo lại collection
        limit: Giới hạn số chunks (0 = toàn bộ)
        batch_size: Batch size khi index
    
    Returns:
        str: Tên collection đã tạo
    """
    print("\n" + "="*80)
    print("STEP 2: BUILD VECTOR STORE")
    print("="*80)
    
    # Build command
    cmd = [
        sys.executable,
        "src/vectorstore/build_vectordb.py",
        "--chunks", chunks_file,
        "--out_dir", qdrant_path,
        "--embedding_model", embedding_model,
        "--batch_size", str(batch_size),
    ]
    
    if collection_name:
        cmd.extend(["--collection", collection_name])
    
    if recreate:
        cmd.append("--recreate")
    
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Chạy vectorstore script
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Extract collection name từ output (nếu auto-generated)
    if not collection_name:
        # Auto-generated: chunks_{name}_{embedding_model_sanitized}
        from vectorstore.build_retriever import sanitize_collection_name
        chunks_basename = os.path.basename(chunks_file).replace(".jsonl", "")
        embedding_safe = sanitize_collection_name(embedding_model)
        collection_name = sanitize_collection_name(f"{chunks_basename}_{embedding_safe}")
    
    print(f"[SUCCESS] Collection created: {collection_name}")
    return collection_name


def get_retriever(
    collection_name: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    search_type: str = "similarity",
    top_k: int = 5,
) -> BaseRetriever:
    """
    Lấy retriever từ Qdrant collection.
    
    Args:
        collection_name: Tên collection trong Qdrant
        qdrant_path: Đường dẫn Qdrant database
        embedding_model: HuggingFace embedding model (phải khớp với khi build)
        search_type: Loại search (similarity, mmr, similarity_score_threshold)
        top_k: Số documents trả về
    
    Returns:
        BaseRetriever: LangChain retriever object
    """
    print("\n" + "="*80)
    print("STEP 3: BUILD RETRIEVER")
    print("="*80)
    
    retriever = build_retriever(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        embedding_model=embedding_model,
        search_type=search_type,
        search_kwargs={"k": top_k},
    )
    
    return retriever


# ==================== EVALUATION FUNCTIONS ====================

def load_evaluation_dataset(
    split: str = "wixqa_expertwritten",
    max_queries: int = 0,
) -> List[Dict[str, Any]]:
    """
    Load WixQA evaluation dataset (QA pairs with ground truth article_ids).
    
    Args:
        split: Dataset split name (default: "wixqa_expertwritten")
        max_queries: Giới hạn số query (0 = toàn bộ)
    
    Returns:
        List[Dict]: List of {question, answer, article_ids}
    """
    print("\n" + "="*80)
    print("LOADING EVALUATION DATASET")
    print("="*80)
    
    dataset = load_dataset("Wix/WixQA", split, split="train")
    
    # Convert to list of dicts
    eval_data = []
    for item in dataset:
        eval_data.append({
            "question": item["question"],
            "answer": item["answer"],
            "article_ids": item["article_ids"],
        })
    
    if max_queries > 0:
        eval_data = eval_data[:max_queries]
    
    print(f"Loaded {len(eval_data)} queries from split: {split}")
    print("="*80)
    
    return eval_data


def aggregate_chunk_scores(
    chunk_scores: List[Tuple[str, float]],
    mode: str = "max",
) -> List[Tuple[str, float]]:
    """
    Gom các chunk scores theo article_id.
    
    Args:
        chunk_scores: List of (article_id, score) từ retrieved chunks
        mode: Aggregation mode - "max", "sum", "mean"
    
    Returns:
        List[(article_id, aggregated_score)] sorted by score descending
    """
    doc_scores = defaultdict(list)
    for article_id, score in chunk_scores:
        doc_scores[article_id].append(score)
    
    # Aggregate scores
    aggregated = []
    for article_id, scores in doc_scores.items():
        if mode == "max":
            agg_score = max(scores)
        elif mode == "sum":
            agg_score = sum(scores)
        elif mode == "mean":
            agg_score = sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        
        aggregated.append((article_id, agg_score))
    
    # Sort by score descending
    aggregated.sort(key=lambda x: x[1], reverse=True)
    
    return aggregated


def evaluate_document_level(
    retriever: BaseRetriever,
    eval_data: List[Dict[str, Any]],
    top_k: int = 5,
    retrieve_k: int = 50,
    agg_mode: str = "max",
) -> Dict[str, float]:
    """
    Hướng 1: Document-level evaluation (gom chunks theo article_id).
    
    Args:
        retriever: LangChain retriever
        eval_data: List of {question, answer, article_ids}
        top_k: K cho metrics @K (số bài đánh giá sau khi gom)
        retrieve_k: Số chunks lấy từ retriever (lấy rộng hơn rồi gom)
        agg_mode: Aggregation mode - "max", "sum", "mean"
    
    Returns:
        Dict[str, float]: Metrics - hit_rate@K, recall@K, precision@K, mrr@K
    """
    print("\n" + "="*80)
    print("EVALUATING: DOCUMENT-LEVEL (CHUNK AGGREGATION)")
    print("="*80)
    print(f"Config: top_k={top_k}, retrieve_k={retrieve_k}, agg_mode={agg_mode}")
    
    hit_scores = []
    recall_scores = []
    precision_scores = []
    rr_scores = []
    
    # Temporarily override retriever's k
    original_k = retriever.search_kwargs.get("k", 5)
    retriever.search_kwargs["k"] = retrieve_k
    
    for item in tqdm(eval_data, desc="Evaluating queries"):
        query = item["question"]
        gold_articles = set(item["article_ids"])
        
        # Retrieve chunks
        retrieved_docs = retriever.invoke(query)
        
        # Extract (article_id, score) from chunks
        chunk_scores = []
        for doc in retrieved_docs:
            article_id = doc.metadata.get("article_id", "")
            # Qdrant similarity search returns docs without scores in metadata
            # We'll use position as inverse score for now
            # Ideally, use similarity_score_threshold or custom retriever
            score = 1.0 / (len(chunk_scores) + 1)  # inverse rank as score
            chunk_scores.append((article_id, score))
        
        # Aggregate chunks -> articles
        ranked_articles = aggregate_chunk_scores(chunk_scores, mode=agg_mode)
        
        # Get top-K articles
        top_k_articles = [aid for aid, _ in ranked_articles[:top_k]]
        
        # Metrics @K
        # Hit Rate @K: 1 nếu có ít nhất 1 bài đúng trong top-K
        hit = 1 if any(aid in gold_articles for aid in top_k_articles) else 0
        hit_scores.append(hit)
        
        # Recall @K: tỷ lệ bài đúng được tìm thấy / tổng số bài đúng
        num_correct_found = len([aid for aid in top_k_articles if aid in gold_articles])
        recall = num_correct_found / len(gold_articles) if gold_articles else 0
        recall_scores.append(recall)
        
        # Precision @K: tỷ lệ bài đúng / K
        precision = num_correct_found / top_k if top_k > 0 else 0
        precision_scores.append(precision)
        
        # MRR @K: 1/rank của bài đúng đầu tiên
        rr = 0
        for rank, aid in enumerate(top_k_articles, start=1):
            if aid in gold_articles:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    
    # Restore original k
    retriever.search_kwargs["k"] = original_k
    
    # Macro-average
    metrics = {
        f"hit_rate@{top_k}": np.mean(hit_scores),
        f"recall@{top_k}": np.mean(recall_scores),
        f"precision@{top_k}": np.mean(precision_scores),
        f"mrr@{top_k}": np.mean(rr_scores),
    }
    
    print("\n" + "-"*80)
    print("DOCUMENT-LEVEL RESULTS:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("-"*80)
    
    return metrics


def build_gold_embeddings(
    article_ids: Set[str],
    collection_name: str,
    retriever: BaseRetriever,
) -> Dict[str, np.ndarray]:
    """
    Build gold embeddings cho mỗi article bằng mean-pooling chunks của article đó.
    
    Args:
        article_ids: Set of article IDs cần tạo gold embeddings
        collection_name: Tên Qdrant collection
        retriever: BaseRetriever với client đã được khởi tạo
    
    Returns:
        Dict[article_id -> gold_embedding_vector]
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    # Lấy client từ vectorstore của retriever
    vectorstore = retriever.vectorstore
    client = vectorstore.client
    
    gold_embeddings = {}
    
    for article_id in tqdm(article_ids, desc="Building gold embeddings"):
        # Scroll all chunks của article này
        # Filter by metadata: article_id
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.article_id",
                        match=MatchValue(value=article_id)
                    )
                ]
            ),
            limit=1000,  # assume không quá 1000 chunks per article
            with_vectors=True,
        )
        
        points = results[0]  # (points, next_page_offset)
        
        if not points:
            continue
        
        # Extract vectors
        vectors = [point.vector for point in points]
        
        # Mean pooling
        mean_vector = np.mean(vectors, axis=0)
        gold_embeddings[article_id] = mean_vector
    
    return gold_embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def evaluate_chunk_level_semantic(
    retriever: BaseRetriever,
    eval_data: List[Dict[str, Any]],
    collection_name: str,
    top_k: int = 5,
    cosine_threshold: float = 0.75,
) -> Dict[str, float]:
    """
    Hướng 2: Chunk-level semantic evaluation (cosine với gold embeddings).
    
    Args:
        retriever: LangChain retriever
        eval_data: List of {question, answer, article_ids}
        collection_name: Tên Qdrant collection (để lấy vectors)
        top_k: K cho metrics @K
        cosine_threshold: Threshold để coi chunk là "correct"
    
    Returns:
        Dict[str, float]: Metrics - hit_rate@K, recall@K, precision@K, mrr@K
    """
    print("\n" + "="*80)
    print("EVALUATING: CHUNK-LEVEL SEMANTIC (COSINE SIMILARITY)")
    print("="*80)
    print(f"Config: top_k={top_k}, cosine_threshold={cosine_threshold}")
    
    # Build gold embeddings for all unique articles in eval_data
    all_gold_articles = set()
    for item in eval_data:
        all_gold_articles.update(item["article_ids"])
    
    print(f"\nBuilding gold embeddings for {len(all_gold_articles)} articles...")
    gold_embeddings = build_gold_embeddings(
        article_ids=all_gold_articles,
        collection_name=collection_name,
        retriever=retriever,
    )
    print(f"[SUCCESS] Built {len(gold_embeddings)} gold embeddings")
    
    # Get embedding function from retriever
    vectorstore = retriever.vectorstore
    embedding_function = vectorstore.embeddings
    
    hit_scores = []
    recall_scores = []
    precision_scores = []
    rr_scores = []
    
    # Temporarily override retriever's k
    original_k = retriever.search_kwargs.get("k", 5)
    retriever.search_kwargs["k"] = top_k
    
    # Debug counters
    debug_stats = {
        "total_chunks": 0,
        "embedded_chunks": 0,
        "max_similarity": 0.0,
        "min_similarity": 1.0,
    }
    
    for item in tqdm(eval_data, desc="Evaluating queries"):
        query = item["question"]
        gold_articles = set(item["article_ids"])
        
        # Retrieve chunks
        retrieved_docs = retriever.invoke(query)
        
        # Get gold embeddings for this query
        gold_vecs = [gold_embeddings.get(aid) for aid in gold_articles if aid in gold_embeddings]
        
        if not gold_vecs:
            # No gold embeddings available, skip
            hit_scores.append(0)
            recall_scores.append(0)
            precision_scores.append(0)
            rr_scores.append(0)
            continue
        
        # Check correctness for each retrieved chunk
        correct_flags = []
        chunk_article_ids = []
        
        for doc in retrieved_docs:
            article_id = doc.metadata.get("article_id", "")
            chunk_article_ids.append(article_id)
            debug_stats["total_chunks"] += 1
            
            # Embed retrieved chunk text to get its vector
            try:
                chunk_text = doc.page_content
                # Embed the text using the same embedding model
                chunk_vec = np.array(embedding_function.embed_query(chunk_text))
                debug_stats["embedded_chunks"] += 1
                
                # Compute max cosine similarity with gold embeddings
                max_sim = max(cosine_similarity(chunk_vec, gold_vec) for gold_vec in gold_vecs)
                
                # Update debug stats
                debug_stats["max_similarity"] = max(debug_stats["max_similarity"], max_sim)
                debug_stats["min_similarity"] = min(debug_stats["min_similarity"], max_sim)
                
                is_correct = max_sim >= cosine_threshold
                correct_flags.append(is_correct)
            except Exception as e:
                # Debug: print first error
                if debug_stats["total_chunks"] == 1:
                    print(f"\n⚠️  Error embedding chunk: {e}")
                correct_flags.append(False)
        
        # Metrics @K (chunk-level semantic)
        # Hit Rate @K: có ít nhất 1 chunk đúng
        hit = 1 if any(correct_flags[:top_k]) else 0
        hit_scores.append(hit)
        
        # Recall @K: số bài đúng được "cover" bởi chunks đúng / tổng số bài đúng
        covered_articles = set()
        for i in range(min(top_k, len(correct_flags))):
            if correct_flags[i]:
                covered_articles.add(chunk_article_ids[i])
        
        recall = len(covered_articles.intersection(gold_articles)) / len(gold_articles) if gold_articles else 0
        recall_scores.append(recall)
        
        # Precision @K: số chunks đúng / K
        precision = sum(correct_flags[:top_k]) / top_k if top_k > 0 else 0
        precision_scores.append(precision)
        
        # MRR @K: 1/rank của chunk đúng đầu tiên
        rr = 0
        for rank in range(min(top_k, len(correct_flags))):
            if correct_flags[rank]:
                rr = 1.0 / (rank + 1)
                break
        rr_scores.append(rr)
    
    # Restore original k
    retriever.search_kwargs["k"] = original_k
    
    # Print debug stats
    print("\nDebug Statistics:")
    print(f"  Total chunks evaluated: {debug_stats['total_chunks']}")
    print(f"  Chunks embedded: {debug_stats['embedded_chunks']}")
    if debug_stats['embedded_chunks'] > 0:
        print(f"  Similarity range: [{debug_stats['min_similarity']:.4f}, {debug_stats['max_similarity']:.4f}]")
        print(f"  Threshold used: {cosine_threshold}")
    
    # Macro-average
    metrics = {
        f"hit_rate@{top_k}_semantic": np.mean(hit_scores),
        f"recall@{top_k}_semantic": np.mean(recall_scores),
        f"precision@{top_k}_semantic": np.mean(precision_scores),
        f"mrr@{top_k}_semantic": np.mean(rr_scores),
    }
    
    print("\n" + "-"*80)
    print("CHUNK-LEVEL SEMANTIC RESULTS:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("-"*80)
    
    return metrics


# ==================== MAIN PIPELINE ====================

def run_full_pipeline(
    strategy: str = "recursive",
    chunk_size: int = 380,
    overlap: int = 50,
    max_records: int = 0,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    recreate: bool = False,
    top_k: int = 5,
) -> BaseRetriever:
    """
    Chạy full pipeline: chunk → vectorstore → retriever.
    
    Args:
        strategy: Chiến lược chunking
        chunk_size: Kích thước chunk
        overlap: Overlap size
        max_records: Giới hạn số bài để chunk (0 = toàn bộ)
        qdrant_path: Đường dẫn Qdrant database
        embedding_model: HuggingFace embedding model
        recreate: Xóa và tạo lại collection
        top_k: Số documents trả về khi retrieve
    
    Returns:
        BaseRetriever: Retriever đã sẵn sàng để evaluate
    """
    print("\n" + "="*80)
    print("FULL PIPELINE: CHUNK -> VECTORSTORE -> RETRIEVER")
    print("="*80)
    
    # Step 1: Chunking
    chunks_file = run_chunking_pipeline(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        max_records=max_records,
    )
    
    # Step 2: Build vector store
    collection_name = run_vectorstore_pipeline(
        chunks_file=chunks_file,
        qdrant_path=qdrant_path,
        embedding_model=embedding_model,
        recreate=recreate,
    )
    
    # Step 3: Get retriever
    retriever = get_retriever(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] FULL PIPELINE COMPLETED")
    print("="*80 + "\n")
    
    return retriever


def run_from_collection(
    collection_name: str,
    qdrant_path: str = "langchain_qdrant",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 5,
) -> BaseRetriever:
    """
    Chạy pipeline từ collection có sẵn: load collection → retriever.
    
    Args:
        collection_name: Tên collection trong Qdrant
        qdrant_path: Đường dẫn Qdrant database
        embedding_model: HuggingFace embedding model
        top_k: Số documents trả về
    
    Returns:
        BaseRetriever: Retriever đã sẵn sàng để evaluate
    """
    print("\n" + "="*80)
    print("LOADING EXISTING COLLECTION")
    print("="*80)
    
    # Get retriever từ collection có sẵn
    retriever = get_retriever(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    
    print("\n" + "="*80)
    print("[SUCCESS] RETRIEVER READY")
    print("="*80 + "\n")
    
    return retriever


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (chunk → vectorstore → retriever)
  python src/evaluate.py --mode full --strategy recursive --chunk_size 380 --overlap 50
  
  # From existing collection
  python src/evaluate.py --mode from_collection --collection chunks_recursive_380_50_baai_bge_small_en_v1_5
        """
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "from_collection"],
        default="full",
        help="Pipeline mode: full (chunk→vectorstore→retriever) hoặc from_collection (load collection→retriever)",
    )
    
    # Chunking params (cho mode=full)
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=["recursive", "token"],
        help="Chiến lược chunking (chỉ dùng với mode=full)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=380,
        help="Kích thước chunk (chỉ dùng với mode=full)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap size (chỉ dùng với mode=full)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Giới hạn số bài để chunk (0=toàn bộ, chỉ dùng với mode=full)",
    )
    
    # Vector store params
    parser.add_argument(
        "--qdrant_path",
        type=str,
        default="langchain_qdrant",
        help="Đường dẫn Qdrant database",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embedding model",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Tên collection (required cho mode=from_collection)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa và tạo lại collection (chỉ dùng với mode=full)",
    )
    
    # Retriever params
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Số documents trả về khi retrieve",
    )
    parser.add_argument(
        "--search_type",
        type=str,
        default="similarity",
        choices=["similarity", "mmr", "similarity_score_threshold"],
        help="Loại search",
    )
    
    # Evaluation params
    parser.add_argument(
        "--eval_method",
        type=str,
        default="both",
        choices=["document", "semantic", "both"],
        help="Phương pháp đánh giá: document (gom chunk), semantic (cosine), both (cả 2)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Giới hạn số queries để đánh giá (0=toàn bộ)",
    )
    parser.add_argument(
        "--retrieve_k",
        type=int,
        default=50,
        help="Số chunks lấy từ retriever cho document-level eval (lấy rộng rồi gom)",
    )
    parser.add_argument(
        "--agg_mode",
        type=str,
        default="max",
        choices=["max", "sum", "mean"],
        help="Aggregation mode cho document-level eval",
    )
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold cho semantic eval",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate args
    if args.mode == "from_collection" and not args.collection:
        print("[ERROR] --collection is required for mode=from_collection")
        sys.exit(1)
    
    # Run pipeline to get retriever
    if args.mode == "full":
        retriever = run_full_pipeline(
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_records=args.max_records,
            qdrant_path=args.qdrant_path,
            embedding_model=args.embedding_model,
            recreate=args.recreate,
            top_k=args.top_k,
        )
        # Auto-determine collection name (same logic as in pipeline)
        from vectorstore.build_retriever import sanitize_collection_name
        chunks_basename = f"chunks_{args.strategy}_{args.chunk_size}_{args.overlap}"
        embedding_safe = sanitize_collection_name(args.embedding_model)
        collection_name = sanitize_collection_name(f"{chunks_basename}_{embedding_safe}")
    else:  # from_collection
        retriever = run_from_collection(
            collection_name=args.collection,
            qdrant_path=args.qdrant_path,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )
        collection_name = args.collection
    
    # Load evaluation dataset
    eval_data = load_evaluation_dataset(max_queries=args.max_queries)
    
    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    all_metrics = {}
    
    # Document-level evaluation
    if args.eval_method in ["document", "both"]:
        doc_metrics = evaluate_document_level(
            retriever=retriever,
            eval_data=eval_data,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            agg_mode=args.agg_mode,
        )
        all_metrics.update(doc_metrics)
    
    # Chunk-level semantic evaluation
    if args.eval_method in ["semantic", "both"]:
        semantic_metrics = evaluate_chunk_level_semantic(
            retriever=retriever,
            eval_data=eval_data,
            collection_name=collection_name,
            top_k=args.top_k,
            cosine_threshold=args.cosine_threshold,
        )
        all_metrics.update(semantic_metrics)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    print(f"Queries evaluated: {len(eval_data)}")
    print(f"Evaluation method: {args.eval_method}")
    print(f"Top-K: {args.top_k}")
    print("\nMetrics:")
    for metric_name, value in all_metrics.items():
        print(f"  {metric_name:30s} {value:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()



