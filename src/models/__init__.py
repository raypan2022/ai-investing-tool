from .openai_client import OpenAIClient
from .fingpt_client import FinGPTClient

def get_model_client(model_type: str = "openai", **kwargs):
    """Factory function to get model client"""
    if model_type == "openai":
        return OpenAIClient(**kwargs)
    elif model_type == "fingpt":
        return FinGPTClient(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
