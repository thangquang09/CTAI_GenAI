from typing import List
from datasets import load_dataset
from .models import KBDoc, QAPair
from ..config import hf_cfg


def load_kb_docs(split: str = "train") -> List[KBDoc]:
    """
    Load WixKB corpus as a list of KBDoc.

    NOTE: You MUST inspect the dataset features once in a notebook:
        from datasets import load_dataset
        d = load_dataset("Wix/WixQA", "wix_kb_corpus", split="train")
        print(d.features)
    and then map the column names here (e.g. 'doc_id', 'title', 'content').
    """
    dataset = load_dataset(
        hf_cfg.dataset_name,
        hf_cfg.kb_config,
        split=split,
    )

    docs: List[KBDoc] = []
    for row in dataset:
        # TODO: adjust these keys to the actual schema
        doc_id = row.get("doc_id") or row.get("id")
        title = row.get("title")
        contents = row.get("contents")
        

        if not doc_id or not contents:
            continue
        docs.append(KBDoc(doc_id=doc_id, title=title, contents=contents))
    return docs


def load_qa_pairs(split: str = "test") -> List[QAPair]:
    """
    Load QA pairs (e.g. from wixqa_expertwritten).
    """
    dataset = load_dataset(
        hf_cfg.dataset_name,
        hf_cfg.qa_config,
        split=split,
    )

    qa_list: List[QAPair] = []
    for row in dataset:
        question = row["question"]
        answer = row["answer"]

        # TODO: adjust the name of ground-truth KB id field; inspect dataset
        kb_ids = (
            row.get("kb_ids")
            or row.get("doc_ids")
            or row.get("positive_kb_ids")
            or row.get("article_ids")
            or []
        )
        qa_list.append(
            QAPair(
                question=question,
                answer=answer,
                kb_ids=list(kb_ids),
            )
        )
    return qa_list
