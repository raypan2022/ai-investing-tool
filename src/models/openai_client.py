import os
from openai import OpenAI


class OpenAIClient:
    """Minimal OpenAI client using Responses API input prompt."""

    def __init__(self, model: str = "gpt-5"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", model)

    def chat(self, input_text: str, instructions: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=input_text,
            instructions=instructions,
        )
        # Prefer SDK convenience
        if getattr(resp, "output_text", None):
            return str(resp.output_text)
        # Fallback to structured output
        try:
            out = getattr(resp, "output", None) or []
            parts = []
            for item in out:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        val = getattr(c, "text", None)
                        if isinstance(val, str) and val:
                            parts.append(val)
                        elif isinstance(c, dict) and isinstance(c.get("text"), str):
                            parts.append(c["text"])
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
        return str(resp)
