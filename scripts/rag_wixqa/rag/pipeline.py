from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document

import sys

sys.path.append("/content/rag_wixqa/src/")
from rag_wixqa.vectorstores.chroma_store import load_chroma_collection


@dataclass
class RAGConfig:
    collection_name: str = "wixqa_recursive_chunks"
    model_name: str = "Qwen/Qwen3-0.6B"
    top_k: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7
    enable_thinking: bool = False  # keep off for simple RAG, avoids <think> parsing

    # You can tweak this prompt later for better performance
    system_prompt: str = (
        "You are a helpful assistant that answers questions about a knowledge base. "
        "Use ONLY the provided context. If the answer is not in the context, say "
        "'I am not sure based on the given documents.'"
    )


class Qwen3RAGPipeline:
    """
    Minimal RAG pipeline:
      - retrieve top-k chunks from Chroma
      - build a chat prompt with context
      - call Qwen3/Qwen3-0.6B via transformers
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # --- Load vector store ---
        self.vectorstore = load_chroma_collection(self.config.collection_name)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.top_k},
        )

        # --- Load Qwen3 model & tokenizer ---
        # Use the HF-recommended pattern for Qwen3 :contentReference[oaicite:1]{index=1}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype="auto",
            device_map="auto",
        )

    # ----------------- RAG steps -----------------

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Document]:
        k = k or self.config.top_k
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        return retriever.invoke(question)

    def _format_context(self, docs: List[Document]) -> str:
        """
        Turn retrieved documents into a single context string.
        """
        parts = []
        for i, d in enumerate(docs, start=1):
            doc_id = d.metadata.get("doc_id", "unknown")
            parts.append(f"[DOC {i} | id={doc_id}]\n{d.page_content}\n")
        return "\n".join(parts)

    def _build_messages(self, question: str, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Build chat messages for Qwen3 using its chat template.

        Qwen3 uses messages like:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        and `tokenizer.apply_chat_template` handles formatting. :contentReference[oaicite:2]{index=2}
        """
        context = self._format_context(docs)
        user_content = (
            "You are given some knowledge base excerpts.\n\n"
            f"{context}\n\n"
            "Answer the following question ONLY using the information in the excerpts. "
            "If you are not sure, say you are not sure.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Call Qwen3 to generate an answer from messages.
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        # Build the chat prompt using Qwen3's chat template.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,  # False by default here
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.8,
                top_k=20,
            )

        # Only keep the newly generated tokens (after the prompt)
        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # If you ever enable thinking mode, you might want to strip <think>...</think> here.
        return output.strip()

    # ----------------- Public API -----------------

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_docs: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entrypoint.

        Returns:
            {
              "answer": str,
              "docs": List[Document]  # optional
            }
        """
        docs = self.retrieve(question, k=top_k)
        messages = self._build_messages(question, docs)
        answer = self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        result: Dict[str, Any] = {"answer": answer}
        if return_docs:
            result["docs"] = docs
        return result
