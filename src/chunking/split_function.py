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

def split_senmatic(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    pass


    