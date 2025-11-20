from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from ..config import emb_cfg


def build_embedding(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible embedding function.
    """
    resolved_model = model_name or emb_cfg.model_name
    return HuggingFaceEmbeddings(model_name=resolved_model)


def get_default_embedding() -> HuggingFaceEmbeddings:
    return build_embedding()
