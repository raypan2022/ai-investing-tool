"""FinGPT client for self-hosted model (RunPod/Modal) with /generate contract.

Only exposes a `generate(user_prompt)` method that posts to
`<endpoint>/generate` and returns the `response` string from the JSON body.
"""

import requests
import os


class FinGPTClient:
    """Client for self-hosted FinGPT model exposing a custom /generate endpoint."""
    
    def __init__(self, 
                 endpoint_url: str = None,
                 model_name: str = "fingpt-7b"):
        """
        Initialize FinGPT client for self-hosted model.
        
        Args:
            endpoint_url: URL of your hosted FinGPT endpoint (Modal, RunPod, etc.)
            model_name: Name of the FinGPT model variant
        """
        # Single source of truth for endpoint configuration
        self.endpoint_url = endpoint_url or os.getenv("RUNPOD_ENDPOINT_URL")
        self.model_name = model_name
        
        if not self.endpoint_url:
            raise ValueError("FinGPT endpoint URL not provided")

    def generate(self, user_prompt: str, *, timeout: int = 60) -> str:
        """Call custom /generate endpoint returning {"response": "..."}."""
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(
                f"{self.endpoint_url}/generate",
                json={"user_prompt": user_prompt},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            return f"Error calling FinGPT /generate: {e}"
