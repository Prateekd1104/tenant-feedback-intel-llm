from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .config import settings

# Simple embedding function (you can plug in a better one, e.g. sentence-transformers)
DEFAULT_EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()


class RAGStore:
    """
    Thin wrapper around ChromaDB for storing policy docs, FAQs, etc.
    """

    def __init__(self, collection_name: str = "tenant_kb"):
        persist_dir = Path(settings.chroma_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=DEFAULT_EMBEDDING_FUNCTION,
        )

    def upsert_docs(self, docs: List[Dict[str, Any]]) -> None:
        """
        docs: list of {id: str, text: str, metadata: dict}
        """
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, text: str, top_k: int = 5) -> List[str]:
        result = self.collection.query(
            query_texts=[text],
            n_results=top_k,
        )
        docs = result.get("documents", [[]])[0]
        return docs
