from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KBDoc:
    doc_id: str
    title: Optional[str]
    contents: str


@dataclass
class QAPair:
    question: str
    answer: str
    kb_ids: List[str]
    article_ids: Optional[List[str]] = None  # optional alias for backwards compatibility

    def __post_init__(self) -> None:
        if self.article_ids is None:
            self.article_ids = self.kb_ids
