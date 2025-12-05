from typing import List, Literal, Dict, Any
from dataclasses import dataclass

import openai  # Used as a generic client interface; configure to hit your LLaMA-3 endpoint.

from .config import settings

# Configure client (this can be adapted to your provider)
openai.api_key = settings.llm_api_key


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]


class LLMClient:
    """
    Simple wrapper around an OpenAI-compatible LLaMA-3 endpoint.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.llm_model_name

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> LLMResponse:
        """
        Generic chat completion call.
        """
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        text = response["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=response)

    def classify_sentiment_emotion(self, text: str) -> Dict[str, Any]:
        """
        Uses LLM to classify sentiment and emotion.

        Returns:
            {
                "sentiment": "positive/negative/neutral",
                "emotion": "anger/frustration/joy/...",
                "confidence": 0-1
            }
        """
        system_prompt = (
            "You are a precise text classification model. "
            "Given a tenant message, you must output a JSON object with keys: "
            "sentiment (positive, negative, neutral), "
            "emotion (frustration, anger, sadness, joy, fear, neutral), "
            "and confidence (a number between 0 and 1). "
            "Return ONLY valid JSON."
        )

        user_prompt = f"Message: {text}"

        res = self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        import json

        try:
            parsed = json.loads(res.text)
        except Exception:
            parsed = {
                "sentiment": "unknown",
                "emotion": "unknown",
                "confidence": 0.5,
            }
        return parsed

    def generate_response_with_context(
        self, tenant_message: str, retrieved_docs: List[str]
    ) -> str:
        """
        RAG-style response generation: we pass retrieved docs as context.
        """
        context_block = "\n\n".join(retrieved_docs)
        system_prompt = (
            "You are a tenant support assistant. "
            "Use ONLY the provided context to draft a clear, polite reply. "
            "If information is missing, ask a clarifying question instead of inventing details."
        )
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Tenant message:\n{tenant_message}\n\n"
            "Draft a helpful response."
        )
        res = self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return res.text
