import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from split_function import split_recursive, split_token


def load_wix_corpus() -> Iterable[Dict[str, Any]]:
    """
    Tải WixQA corpus (wix_kb_corpus) chỉ lấy các field cần thiết.
    """
    ds = load_dataset("Wix/WixQA", "wix_kb_corpus", split="train")
    for rec in ds:
        yield {
            "id": rec["id"],
            "title": rec.get("title"),
            "url": rec.get("url"),
            "article_type": rec.get("article_type"),
            "contents": rec.get("contents") or "",
        }


# Code pipeline chunk chính
def make_chunks_for_records(
    rec: Dict[str, Any],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    text = rec.get("contents") or ""
    if not text.strip():
        return []

    if strategy == "recursive":
        parts = split_recursive(text, chunk_size, chunk_overlap)
    elif strategy == "token":
        parts = split_token(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        # thông số semantic có thể chỉnh trong hàm; chunk_size/overlap dùng cho nén hậu kì
        # parts = split_semantic(text, max_chunk_size=chunk_size)
        pass
    elif strategy == "hybrid_para_token":
        # parts = split_hybrid_para_token(text, chunk_size, chunk_overlap)
        pass
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    out = []
    for i, p in enumerate(parts):
        if not p.strip():
            continue
        out.append(
            {
                "chunk_id": f"{rec['id']}::{i}",
                "article_id": rec["id"],
                "title": rec.get("title"),
                "url": rec.get("url"),
                "article_type": rec.get("article_type"),
                "text": p,
                # "n_tokens": count_tokens(p),
                "position": i,
            }
        )
    return out


def parse_arg():
    ap = argparse.ArgumentParser(
        description="Chunk WixQA corpus.contents với LangChain nhiều chiến lược"
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="token",
        choices=["recursive", "token", "semantic", "hybrid_para_token"],
        help="Chiến lược chunk",
    )
    ap.add_argument(
        "--chunk_size", type=int, default=380, help="Kích thước chunk (token)"
    )
    ap.add_argument("--overlap", type=int, default=50, help="Số token overlap")
    ap.add_argument(
        "--max_records", type=int, default=0, help="Giới hạn số bài (0 = toàn bộ)"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Đường dẫn file JSONL output",
    )
    args = ap.parse_args()

    return args


def main():
    args = parse_arg()
    
    if args.out == "":
        out_dir = f"data/chunks/chunks_{args.strategy}_{args.chunk_size}_{args.overlap}.jsonl"
    else:
        out_dir = args.out
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)

    n = 0
    total_chunks = 0

    with open(out_dir, "w", encoding="utf-8") as f:
        for rec in load_wix_corpus():
            chunks = make_chunks_for_records(
                rec,
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
            )
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            n += 1
            if args.max_records and n >= args.max_records:
                break

    print(
        f"Done. Articles processed: {n} | Chunks written: {total_chunks} → {out_dir}"
    )


if __name__ == "__main__":
    main()
