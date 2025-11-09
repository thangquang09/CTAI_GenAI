from typing import Any, Dict, Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


# Define các hàm split khác nhau
def split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text or "")


def split_token(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_text(text or "")


def split_semantic(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str = "BAAI/bge-m3",
    threshold_type: str = "percentile",  # 'percentile' | 'standard_deviation' | 'interquartile' | 'gradient'
    threshold_amount: float = 95,  # càng cao → ít cắt hơn (chunk dài, “gộp” nhiều hơn)
) -> List[str]:
    """
    Tách theo ngữ nghĩa với LangChain SemanticChunker, rồi nén hậu kỳ theo token.
    """
    if not text or not text.strip():
        return []

    try:
        from langchain_text_splitters import SemanticChunker
    except Exception as e:
        raise ImportError(
            "SemanticChunker chưa sẵn sàng. Cài: "
            "pip install -U langchain-experimental langchain-huggingface sentence-transformers"
        ) from e

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},  # cosine ổn định hơn
    )

    # Tách theo câu + gộp các câu tương đồng trong không gian embedding
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
        # Regex tách câu (có thể tinh chỉnh thêm cho tiếng Việt nếu cần)
        sentence_split_regex=r"(?<=[\.\?\!])\s+",
    )
    semantic_chunks = chunker.split_text(text)

    # Nén hậu kỳ để đảm bảo budget token thống nhất
    if chunk_size and chunk_size > 0:
        tok = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        final_chunks: List[str] = []
        for c in semantic_chunks:
            final_chunks.extend(tok.split_text(c))
        return final_chunks

    return semantic_chunks
