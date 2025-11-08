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
    config_name: str = "wixqa_expertwritten",
    max_queries: int = 0,
) -> List[Dict[str, Any]]:
    """
    Load WixQA evaluation dataset (QA pairs with ground truth article_ids).
    
    Args:
        config_name: Dataset config name (wixqa_expertwritten or wixqa_simulated)
        max_queries: Giới hạn số query (0 = toàn bộ)
    
    Returns:
        List[Dict]: List of {question, answer, article_ids}
    """
    print("\n" + "="*80)
    print("LOADING EVALUATION DATASET")
    print("="*80)
    
    dataset = load_dataset("Wix/WixQA", config_name, split="train")
    
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
    
    print(f"Loaded {len(eval_data)} queries from config: {config_name}")
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


def compute_ndcg_at_k(
    ranked_articles: List[str],
    gold_articles: Set[str],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @K.
    
    Args:
        ranked_articles: List of article IDs in ranked order
        gold_articles: Set of ground truth article IDs
        k: Cutoff position
    
    Returns:
        float: nDCG@K score
    """
    # DCG@K
    dcg = 0.0
    for i, article_id in enumerate(ranked_articles[:k], start=1):
        rel = 1 if article_id in gold_articles else 0
        dcg += rel / np.log2(i + 1)
    
    # IDCG@K (ideal DCG)
    ideal_k = min(k, len(gold_articles))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_k + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


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
        Dict[str, float]: Metrics - hit_rate@K, recall@K, precision@K, mrr@K, ndcg@K, coverage@K
    """
    print("\n" + "="*80)
    print("EVALUATING: DOCUMENT-LEVEL (CHUNK AGGREGATION)")
    print("="*80)
    print(f"Config: top_k={top_k}, retrieve_k={retrieve_k}, agg_mode={agg_mode}")
    
    hit_scores = []
    recall_scores = []
    precision_scores = []
    rr_scores = []
    ndcg_scores = []
    coverage_scores = []
    
    # Get vectorstore to use similarity_search_with_score
    vectorstore = retriever.vectorstore
    
    for item in tqdm(eval_data, desc="Evaluating queries"):
        query = item["question"]
        gold_articles = set(item["article_ids"])
        
        # Retrieve chunks with REAL similarity scores
        retrieved_docs_with_scores = vectorstore.similarity_search_with_score(
            query, k=retrieve_k
        )
        
        # Extract (article_id, score) from chunks
        chunk_scores = []
        for doc, score in retrieved_docs_with_scores:
            article_id = doc.metadata.get("article_id", "")
            if article_id:  # Skip if no article_id
                chunk_scores.append((article_id, score))
        
        # Aggregate chunks -> articles
        ranked_articles = aggregate_chunk_scores(chunk_scores, mode=agg_mode)
        
        # Get top-K articles
        top_k_articles = [aid for aid, _ in ranked_articles[:top_k]]
        
        # Coverage@K: số gold articles được tìm thấy
        covered_articles = set(top_k_articles).intersection(gold_articles)
        coverage = len(covered_articles)
        coverage_scores.append(coverage)
        
        # Hit Rate @K: 1 nếu có ít nhất 1 bài đúng trong top-K
        hit = 1 if coverage > 0 else 0
        hit_scores.append(hit)
        
        # Recall @K: tỷ lệ bài đúng được tìm thấy / tổng số bài đúng
        recall = coverage / len(gold_articles) if gold_articles else 0
        recall_scores.append(recall)
        
        # Precision @K: tỷ lệ bài đúng / K
        precision = coverage / top_k if top_k > 0 else 0
        precision_scores.append(precision)
        
        # MRR @K: 1/rank của bài đúng đầu tiên
        rr = 0
        for rank, aid in enumerate(top_k_articles, start=1):
            if aid in gold_articles:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
        
        # nDCG @K
        ndcg = compute_ndcg_at_k(top_k_articles, gold_articles, top_k)
        ndcg_scores.append(ndcg)
    
    # Macro-average
    metrics = {
        f"hit_rate@{top_k}": np.mean(hit_scores),
        f"recall@{top_k}": np.mean(recall_scores),
        f"precision@{top_k}": np.mean(precision_scores),
        f"mrr@{top_k}": np.mean(rr_scores),
        f"ndcg@{top_k}": np.mean(ndcg_scores),
        f"coverage@{top_k}": np.mean(coverage_scores),
    }
    
    print("\n" + "-"*80)
    print("DOCUMENT-LEVEL RESULTS:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("-"*80)
    
    return metrics


def detect_metadata_key(
    collection_name: str,
    client,
) -> str:
    """
    Tự động phát hiện metadata key structure trong Qdrant collection.
    
    Args:
        collection_name: Tên collection
        client: Qdrant client
    
    Returns:
        str: "metadata.article_id" hoặc "article_id"
    """
    # Lấy 1 point bất kỳ để kiểm tra cấu trúc
    results = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
    )
    
    if not results[0]:
        raise ValueError(f"Collection {collection_name} is empty!")
    
    point = results[0][0]
    payload = point.payload
    
    # Kiểm tra cấu trúc
    if "metadata" in payload and "article_id" in payload.get("metadata", {}):
        return "metadata.article_id"
    elif "article_id" in payload:
        return "article_id"
    else:
        raise ValueError(f"Cannot find article_id in payload structure: {payload.keys()}")


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
    
    # Auto-detect metadata key structure
    print("Detecting metadata key structure...")
    metadata_key = detect_metadata_key(collection_name, client)
    print(f"Using filter key: {metadata_key}")
    
    gold_embeddings = {}
    
    for article_id in tqdm(article_ids, desc="Building gold embeddings"):
        # Scroll all chunks của article này
        # Filter by metadata: article_id
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=metadata_key,
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
        Dict[str, float]: Metrics - hit_rate@K, recall@K, precision@K, mrr@K, ndcg@K, coverage@K
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
    
    # Get vectorstore and client
    vectorstore = retriever.vectorstore
    client = vectorstore.client
    
    hit_scores = []
    recall_scores = []
    precision_scores = []
    rr_scores = []
    ndcg_scores = []
    coverage_scores = []
    
    # Debug counters
    debug_stats = {
        "total_chunks": 0,
        "retrieved_vectors": 0,
        "max_similarity": 0.0,
        "min_similarity": 1.0,
        "similarity_distribution": [],
    }
    
    for item in tqdm(eval_data, desc="Evaluating queries"):
        query = item["question"]
        gold_articles = set(item["article_ids"])
        
        # Retrieve chunks with scores (để lấy được chunk IDs)
        retrieved_docs_with_scores = vectorstore.similarity_search_with_score(
            query, k=top_k
        )
        
        # Get gold embeddings for this query
        gold_vecs = [gold_embeddings.get(aid) for aid in gold_articles if aid in gold_embeddings]
        
        if not gold_vecs:
            # No gold embeddings available, skip
            hit_scores.append(0)
            recall_scores.append(0)
            precision_scores.append(0)
            rr_scores.append(0)
            ndcg_scores.append(0)
            coverage_scores.append(0)
            continue
        
        # Check correctness for each retrieved chunk
        correct_flags = []
        chunk_article_ids = []
        
        for doc, score in retrieved_docs_with_scores:
            article_id = doc.metadata.get("article_id", "")
            chunk_id = doc.metadata.get("chunk_id", "")
            chunk_article_ids.append(article_id)
            debug_stats["total_chunks"] += 1
            
            try:
                # Lấy vector từ Qdrant thay vì re-embed
                if chunk_id:
                    # Retrieve vector by chunk_id (UUID)
                    point = client.retrieve(
                        collection_name=collection_name,
                        ids=[chunk_id],
                        with_vectors=True,
                    )
                    
                    if point and len(point) > 0:
                        chunk_vec = np.array(point[0].vector)
                        debug_stats["retrieved_vectors"] += 1
                    else:
                        # Fallback: re-embed nếu không lấy được vector
                        chunk_text = doc.page_content
                        chunk_vec = np.array(vectorstore.embeddings.embed_query(chunk_text))
                else:
                    # No chunk_id, fallback to re-embed
                    chunk_text = doc.page_content
                    chunk_vec = np.array(vectorstore.embeddings.embed_query(chunk_text))
                
                # Compute max cosine similarity with gold embeddings
                max_sim = max(cosine_similarity(chunk_vec, gold_vec) for gold_vec in gold_vecs)
                
                # Update debug stats
                debug_stats["max_similarity"] = max(debug_stats["max_similarity"], max_sim)
                debug_stats["min_similarity"] = min(debug_stats["min_similarity"], max_sim)
                debug_stats["similarity_distribution"].append(max_sim)
                
                is_correct = max_sim >= cosine_threshold
                correct_flags.append(is_correct)
            except Exception as e:
                # Debug: print first error
                if debug_stats["total_chunks"] == 1:
                    print(f"\n⚠️  Error retrieving/embedding chunk: {e}")
                correct_flags.append(False)
        
        # Coverage@K: số gold articles được "cover" bởi chunks đúng
        covered_articles = set()
        for i in range(min(top_k, len(correct_flags))):
            if correct_flags[i]:
                covered_articles.add(chunk_article_ids[i])
        
        coverage = len(covered_articles.intersection(gold_articles))
        coverage_scores.append(coverage)
        
        # Metrics @K (chunk-level semantic)
        # Hit Rate @K: có ít nhất 1 chunk đúng
        hit = 1 if any(correct_flags[:top_k]) else 0
        hit_scores.append(hit)
        
        # Recall @K: số bài đúng được "cover" bởi chunks đúng / tổng số bài đúng
        recall = coverage / len(gold_articles) if gold_articles else 0
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
        
        # nDCG @K: treat correct chunks as relevant (binary relevance)
        # Convert correct_flags to ranked article list for nDCG
        ranked_by_correctness = [
            chunk_article_ids[i] for i in range(min(top_k, len(correct_flags)))
            if correct_flags[i]
        ]
        ndcg = compute_ndcg_at_k(ranked_by_correctness, gold_articles, top_k)
        ndcg_scores.append(ndcg)
    
    # Print debug stats
    print("\nDebug Statistics:")
    print(f"  Total chunks evaluated: {debug_stats['total_chunks']}")
    print(f"  Vectors retrieved from Qdrant: {debug_stats['retrieved_vectors']}")
    print(f"  Re-embedded chunks: {debug_stats['total_chunks'] - debug_stats['retrieved_vectors']}")
    if debug_stats['retrieved_vectors'] > 0:
        similarities = debug_stats['similarity_distribution']
        print(f"  Similarity range: [{debug_stats['min_similarity']:.4f}, {debug_stats['max_similarity']:.4f}]")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Median similarity: {np.median(similarities):.4f}")
        print(f"  Std similarity: {np.std(similarities):.4f}")
        print(f"  Threshold used: {cosine_threshold}")
        
        # Percentiles for threshold calibration
        percentiles = [25, 50, 75, 90, 95]
        print("\n  Similarity percentiles (for threshold calibration):")
        for p in percentiles:
            val = np.percentile(similarities, p)
            print(f"    {p}th percentile: {val:.4f}")
        
        # Proportion above threshold
        above_threshold = sum(1 for s in similarities if s >= cosine_threshold) / len(similarities)
        print(f"\n  Chunks above threshold ({cosine_threshold}): {above_threshold:.2%}")
    
    # Macro-average
    metrics = {
        f"hit_rate@{top_k}_semantic": np.mean(hit_scores),
        f"recall@{top_k}_semantic": np.mean(recall_scores),
        f"precision@{top_k}_semantic": np.mean(precision_scores),
        f"mrr@{top_k}_semantic": np.mean(rr_scores),
        f"ndcg@{top_k}_semantic": np.mean(ndcg_scores),
        f"coverage@{top_k}_semantic": np.mean(coverage_scores),
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



