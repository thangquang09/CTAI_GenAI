from pathlib import Path
from pydantic import BaseModel


class Paths(BaseModel):
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw" / "wixqa"
    data_processed: Path = root / "data" / "processed"
    chroma_root: Path = root / "data" / "chroma"


class HFConfigs(BaseModel):
    dataset_name: str = "Wix/WixQA"
    kb_config: str = "wix_kb_corpus"          # KB snapshot config :contentReference[oaicite:2]{index=2}
    qa_config: str = "wixqa_expertwritten"    # you can also try "wixqa_simulated", "wixqa_synthetic"


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384


class ChunkingConfig(BaseModel):
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64


paths = Paths()
hf_cfg = HFConfigs()
emb_cfg = EmbeddingConfig()
chunk_cfg = ChunkingConfig()


if __name__ == "__main__":
    print("paths.root", paths.root)
    print("paths.data_raw ", paths.data_raw)
    print("paths.data_processed ", paths.data_processed)
    print("paths.chroma_root ", paths.chroma_root)
