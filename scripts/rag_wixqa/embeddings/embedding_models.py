from langchain_huggingface import HuggingFaceEmbeddings
from ..config import emb_cfg


def get_default_embedding():
    """
    Returns a LangChain-compatible embedding function using
    sentence-transformers.
    """
    return HuggingFaceEmbeddings(model_name=emb_cfg.model_name)
