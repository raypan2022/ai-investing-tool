import os
from typing import Dict, Any, List
from openai import OpenAI


class OpenAIClient:
    """Minimal OpenAI chat client."""

    def __init__(self, model: str = "gpt-5"):
        # API key resolution handled by the SDK; explicit for clarity
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", model)
        # Optional "thinking"/reasoning control for models that support it (o3/gpt‑4.1/…)
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "medium")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_completion_tokens: int = 1200,
    ) -> str:
        """Send chat request using the Responses API with thinking mode; fallback to Chat Completions only if needed."""
        # Prefer Responses API
        try:
            # Flatten messages for a simple single-input text prompt
            input_text = "\n\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
            resp = self.client.responses.create(
                model=self.model,
                input=input_text,
                temperature=temperature,
                max_output_tokens=max_completion_tokens,
                reasoning={"effort": self.reasoning_effort},
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text
            try:
                return resp.output[0].content[0].text  # type: ignore[attr-defined]
            except Exception:
                return str(resp)
        except Exception as e_resp:
            # Fallback path: Chat Completions (for older models/environments)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e_chat:
                return f"Error from OpenAI: {e_resp or e_chat}"

    def analyze_with_context(self, ticker: str, context: str, question: str) -> str:
        """Convenience wrapper for single-turn Q&A with provided context."""
        system = (
            "You are a precise financial analyst. Use the provided context to answer."
        )
        user = f"Ticker: {ticker}\n\nContext:\n{context}\n\nQuestion:\n{question}"
        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])