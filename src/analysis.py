from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.cluster import KMeans

from .models_llm import LLMClient
from .rag_pipeline import RAGStore


class FeedbackAnalyzer:
    """
    Main orchestrator for analyzing tenant feedback and generating responses.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.rag_store = RAGStore()

    def analyze_single(
        self,
        text: str,
        channel: str,
        tenant_id: Optional[str] = None,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single tenant message and generate an auto-response.
        """

        # 1. Sentiment & emotion
        se = self.llm.classify_sentiment_emotion(text)

        # 2. Retrieve context for RAG
        retrieved_docs = self.rag_store.query(text, top_k=3)

        # 3. Auto-generate response
        response = self.llm.generate_response_with_context(
            tenant_message=text,
            retrieved_docs=retrieved_docs,
        )

        # 4. Build a root-cause heuristic (very simple, you can enhance)
        root_cause = self._infer_root_cause(text)

        result = {
            "channel": channel,
            "tenant_id": tenant_id,
            "context_id": context_id,
            "sentiment": se.get("sentiment"),
            "emotion": se.get("emotion"),
            "sentiment_confidence": se.get("confidence"),
            "root_cause": root_cause,
            "retrieved_docs": retrieved_docs,
            "auto_response": response,
        }
        return result

    @staticmethod
    def _infer_root_cause(text: str) -> str:
        """
        Very simple heuristic; in a more advanced version this can be
        a separate classifier or unsupervised keyword extraction.
        """
        text_lower = text.lower()
        if "leak" in text_lower or "water" in text_lower:
            return "Maintenance issue"
        if "rent" in text_lower or "payment" in text_lower:
            return "Billing / payment issue"
        if "noise" in text_lower:
            return "Noise / neighbour complaint"
        return "General service issue"

    # --- Optional: simple clustering logic for a batch of messages ---

    def cluster_themes(self, texts: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Clusters texts into themes using embeddings from Chroma's embedding function.
        (Note: here we reuse the RAG embedding function implicitly via Chroma client.)
        """
        # For simplicity, we use the same embedding function as RAGStore
        from chromadb.utils import embedding_functions

        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        vectors = embedding_fn(texts)
        X = np.array(vectors)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[int(label)].append(texts[idx])

        return {"clusters": clusters}
