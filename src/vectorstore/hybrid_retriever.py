"""
Hybrid Retriever combining Dense (Vector) + Sparse (BM25) search.

Uses Reciprocal Rank Fusion (RRF) to merge results from:
1. Dense retrieval: Vector similarity search (existing Qdrant)
2. Sparse retrieval: BM25 keyword search

This provides better retrieval by combining semantic and keyword matching.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) search.
    
    Attributes:
        dense_retriever: Vector-based retriever (Qdrant)
        bm25: BM25 index for keyword search
        documents: All documents for BM25 search
        doc_id_to_doc: Mapping from chunk_id to Document
        k: Number of documents to retrieve
        alpha: Weight for dense vs sparse (0=sparse only, 1=dense only, 0.5=equal)
        rrf_k: RRF k parameter (default=60)
    """
    
    dense_retriever: BaseRetriever
    bm25: Any  # BM25Okapi instance
    documents: List[Document]
    doc_id_to_doc: Dict[str, Document]
    k: int = 5
    alpha: float = 0.5  # Weight: 0=sparse only, 1=dense only
    rrf_k: int = 60  # RRF parameter
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (dense + sparse with RRF).
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of top-k documents ranked by RRF fusion score
        """
        # 1. Dense retrieval (vector search)
        dense_docs = self.dense_retriever.invoke(query)
        
        # 2. Sparse retrieval (BM25)
        sparse_docs = self._bm25_search(query, k=self.k * 2)  # Retrieve more for fusion
        
        # 3. Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(
            dense_results=dense_docs,
            sparse_results=sparse_docs,
            k=self.k,
            alpha=self.alpha,
            rrf_k=self.rrf_k,
        )
        
        return fused_docs
    
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """
        BM25 keyword search.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Top-k documents by BM25 score
        """
        # Tokenize query (simple whitespace split)
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        # Return corresponding documents
        return [self.documents[i] for i in top_k_indices]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        k: int,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
        
        RRF score for document d:
            score(d) = alpha * sum(1/(rrf_k + rank_dense(d))) + 
                      (1-alpha) * sum(1/(rrf_k + rank_sparse(d)))
        
        Args:
            dense_results: Results from dense retriever (ranked)
            sparse_results: Results from sparse retriever (ranked)
            k: Number of final documents to return
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            rrf_k: RRF constant (typically 60)
            
        Returns:
            Top-k documents by RRF score
        """
        # Build rank mappings
        dense_ranks = {
            doc.metadata.get("chunk_id"): rank
            for rank, doc in enumerate(dense_results, start=1)
        }
        sparse_ranks = {
            doc.metadata.get("chunk_id"): rank
            for rank, doc in enumerate(sparse_results, start=1)
        }
        
        # Compute RRF scores
        rrf_scores = defaultdict(float)
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        for chunk_id in all_chunk_ids:
            # Dense contribution
            if chunk_id in dense_ranks:
                dense_score = 1.0 / (rrf_k + dense_ranks[chunk_id])
            else:
                dense_score = 0.0
            
            # Sparse contribution
            if chunk_id in sparse_ranks:
                sparse_score = 1.0 / (rrf_k + sparse_ranks[chunk_id])
            else:
                sparse_score = 0.0
            
            # Weighted RRF score
            rrf_scores[chunk_id] = alpha * dense_score + (1 - alpha) * sparse_score
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )
        
        # Get top-k documents
        top_k_docs = []
        for chunk_id in sorted_chunk_ids[:k]:
            if chunk_id in self.doc_id_to_doc:
                top_k_docs.append(self.doc_id_to_doc[chunk_id])
        
        return top_k_docs


def build_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Build BM25 index from documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        BM25Okapi index
    """
    # Tokenize documents (simple whitespace split)
    tokenized_corpus = [
        doc.page_content.lower().split()
        for doc in documents
    ]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def build_hybrid_retriever(
    dense_retriever: BaseRetriever,
    documents: List[Document],
    k: int = 5,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> HybridRetriever:
    """
    Build hybrid retriever from dense retriever and document corpus.
    
    Args:
        dense_retriever: Existing vector-based retriever (e.g., from Qdrant)
        documents: All documents in corpus (for BM25)
        k: Number of documents to retrieve
        alpha: Weight for dense vs sparse (0=sparse only, 1=dense only, 0.5=equal)
        rrf_k: RRF constant parameter
        
    Returns:
        HybridRetriever instance
        
    Example:
        >>> from vectorstore.build_retriever import build_retriever
        >>> from vectorstore.hybrid_retriever import build_hybrid_retriever
        >>> 
        >>> # Build dense retriever
        >>> dense_retriever = build_retriever(
        ...     collection_name="chunks_recursive_380_50_baai_bge_small_en_v1_5",
        ...     search_kwargs={"k": 10}
        ... )
        >>> 
        >>> # Load all documents (for BM25)
        >>> docs = load_documents_from_jsonl("data/chunks/chunks_recursive_380_50.jsonl")
        >>> 
        >>> # Build hybrid retriever
        >>> hybrid_retriever = build_hybrid_retriever(
        ...     dense_retriever=dense_retriever,
        ...     documents=docs,
        ...     k=5,
        ...     alpha=0.5  # Equal weight
        ... )
        >>> 
        >>> # Use hybrid retriever
        >>> results = hybrid_retriever.invoke("How to create a Wix event?")
    """
    # Build BM25 index
    print(f"Building BM25 index for {len(documents)} documents...")
    bm25 = build_bm25_index(documents)
    
    # Build chunk_id -> Document mapping
    doc_id_to_doc = {
        doc.metadata.get("chunk_id"): doc
        for doc in documents
    }
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25=bm25,
        documents=documents,
        doc_id_to_doc=doc_id_to_doc,
        k=k,
        alpha=alpha,
        rrf_k=rrf_k,
    )
    
    print("✅ Hybrid retriever ready!")
    print(f"   Dense retriever: {type(dense_retriever).__name__}")
    print(f"   BM25 corpus size: {len(documents)}")
    print(f"   k={k}, alpha={alpha} (0=sparse, 1=dense), rrf_k={rrf_k}")
    
    return hybrid_retriever
