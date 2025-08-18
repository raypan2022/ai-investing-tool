import os
import pytest
from dotenv import load_dotenv

from src.models.fingpt_client import FinGPTClient


def test_runpod_generate_real_call_prints_response(capsys):
    """Integration test: loads RUNPOD_ENDPOINT_URL from .env and calls endpoint for real.

    Skips if RUNPOD_ENDPOINT_URL is not set.
    """
    load_dotenv()
    endpoint = os.getenv("RUNPOD_ENDPOINT_URL")
    if not endpoint:
        pytest.skip("RUNPOD_ENDPOINT_URL not set; skipping real endpoint test")

    user_prompt = (
        "[Company Introduction]:\n\n"
        "NVIDIA is a leading entity in the Semiconductors sector. Incorporated and publicly traded since 1999, the company has established its reputation as one of the key players in the market. As of today, NVIDIA has a market capitalization of 1500000000000.00 in USD, with 2470000000.00 shares outstanding. NVIDIA operates primarily in the United States, trading under the ticker NVDA on the NASDAQ. As a dominant force in the Semiconductors space, the company continues to innovate and drive progress within the industry.\n\n"
        "From 2023-08-01 to 2023-08-31, NVIDIA's stock price increased from 450.00 to 500.00. Company news during this period are listed below:\n\n"
        "[Headline]: NVIDIA Releases New AI GPU\n"
        "[Summary]: The company released the H200 GPU, promising 50% better performance on LLM workloads.\n\n"
        "[Headline]: Strong Q2 Earnings\n"
        "[Summary]: Revenue rose 70% YoY, exceeding Wall Street expectations.\n\n"
        "Some recent basic financials of NVIDIA, reported at 2023-08-15, are presented below:\n\n"
        "[Basic Financials]:\n"
        "P/E Ratio: 95\n"
        "EPS: 5.20\n"
        "Gross Margin: 68%\n\n"
        "[Technical Analysis]:\n"
        "• RSI: 74 (suggesting overbought conditions)\n"
        "• MACD shows a bullish crossover\n"
        "• 50-day MA crossed above 200-day MA (Golden Cross)\n"
        "• Volume surged 2x on earnings release day\n\n"
        "Based on all the information before 2023-09-01, let's first analyze the positive developments and potential concerns for NVDA. "
        "Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. "
        "Then make your prediction of the NVDA stock price movement for next week (2023-09-08). Provide a summary analysis to support your prediction."
    )

    client = FinGPTClient()
    print(f"\n[RunPod endpoint]: {client.endpoint_url}")
    print(f"[Prompt length]: {len(user_prompt)} characters")
    print(f"[Prompt]: {user_prompt}")
    out = client.generate(user_prompt=user_prompt)
    print("[RunPod /generate response]:\n" + str(out))
    print(f"[Response length]: {len(out or '')}")

    assert isinstance(out, str)
    assert len(out.strip()) > 0
    assert not out.startswith("Error")

