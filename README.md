## AI Investing Tool — Data-Backed Recommendations and Research Assistant

### Overview

This project builds a stock analysis platform that delivers data-backed recommendations and a research chat assistant. It combines:

- Short-horizon forecasting (FinGPT forecaster service)
- Real-time stock basics (yfinance)
- Technical signals (pandas-ta)
- Financial news (Finnhub)
- Retrieval over SEC filings (RAG with FAISS + sentence-transformers)

The recommendation workflow composes these tools into a concise prompt for FinGPT, and optionally performs a second-stage synthesis using RAG citations.

### Forecasting and synthesis pipeline (FinGPT → RAG/OpenAI)

Why this design

- Separation of concerns: FinGPT focuses on short-horizon directionality; OpenAI focuses on cited explanation and decision framing.
- Better signal-to-noise: the forecaster receives only compact, high-signal inputs; long filings and verbose context are moved to the synthesis stage.
- Explainability: RAG over SEC filings surfaces risks/opportunities with citations; users see “why,” not just a number.

Stage 1: FinGPT forecaster (short-horizon view)

- Inputs (curated and compact):
  - Company basics (StockClient): ticker, company name, sector/industry, price, market cap, 52w range, 1d change.
  - Company headlines (NewsClient.company): 2–3 items (headline, summary, source) that explicitly match the ticker/name.
  - Macro headlines (NewsClient.market): 2–3 items (headline, summary, source) filtered by macro keywords and screened for single-stock noise.
  - Technical snapshot (TechnicalAnalyzer): minimal indicators and a synthesized verdict+confidence.
- Prompting constraints:
  - Keep sections concise (one-liners where possible).
  - Avoid pasting filings or long text; summarize instead.
  - Freeze date windows (from/to) for determinism when needed.
- Output: direction (up/down/flat) for next short window, rationale snippets, confidence.

Model details (FinGPT Forecaster)

- The forecaster we use is a PEFT-based adapter (LoRA) fine-tuned on Llama‑2‑7B‑Chat, published on Hugging Face as `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora` [link](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora). It is loaded by applying the PEFT adapter onto the base chat model and then served behind our FastAPI endpoint (e.g., on RunPod).

Stage 2: OpenAI synthesis with RAG (explainability and decision)

- Retrieval:
  - Build/maintain FAISS index over SEC filings (10-K/10-Q/8-K) using sentence-transformers.
  - Retrieve top-k chunks for themes: guidance, liquidity, leverage, litigation, outlook.
  - Extract 3–5 bullets each for positives and concerns, with citations (filing + section/page).
- Synthesis prompt (to OpenAI):
  - Provide: (a) FinGPT forecast output, (b) condensed inputs (one-liners for basics/news/technicals), and (c) RAG bullets with citations.
  - Ask: 3–4 positives and 3–4 concerns; a probability-weighted next-week view; Buy/Hold/Sell with rationale and explicit risks; include citations.
- Output: a user-facing, cited recommendation and brief rationale aligned to the FinGPT direction (or highlighting divergence).

Optional scoring layer (aggregator)

- Combine multiple signals into a single score for thresholding:
  - Forecaster confidence (signed by direction)
  - Technical overall_signal × confidence
  - News breadth/recency (e.g., number of macro/company hits in the window)
  - Filing risk flags (penalize material risk bullets)
- Map thresholds to Buy/Hold/Sell; the OpenAI summary provides the narrative and citations.

Operational notes

- Cache upstream responses and set explicit date windows for reproducibility.
- Keep forecaster prompts small; rely on synthesis for depth.
- Handle errors at the facade; show graceful fallbacks in the UI.

### Architecture

```
Streamlit UI
   ↓
Tools Facade (src/tools/tools.py)
   ├── get_stock_data()        # StockClient (yfinance)
   ├── analyze_technical_*()   # TechnicalAnalyzer (pandas-ta)
   └── get_news()              # NewsClient (Finnhub)
   ↓
Prompt Builder → FinGPT Forecaster API (RunPod/FASTAPI)
   ↓
Optional: RAG over SEC filings (FAISS + sentence-transformers) → OpenAI synthesis
   ↓
Recommendation with rationale and citations
```

### Key modules

- Stock data: `src/tools/stock_data.py`

  - Why: Stable fundamentals keep the forecaster grounded (market cap scale, valuation, sector/industry context) and reduce hallucinations.
  - How: Merge fast price/volume with identity metadata; normalize types; trim descriptions; add UTC timestamps. History uses adjusted daily bars to avoid split/dividend distortions.

- Technical analysis: `src/tools/technical.py`

  - Why: Short-horizon movement is driven by trend/momentum/volatility/flow; encoding these explicitly gives the forecaster signal without noise.
  - How: Minimal set (RSI, MACD, SMA20/50/200, Bollinger, ATR, volume ratio) plus pivots/nearest levels. Rule-based buy/sell tally yields `overall_signal` and bounded `confidence` for compact prompting.

- News: `src/tools/news.py`

  - Why: Combining company-specific headlines (idiosyncratic catalysts) with macro headlines (policy/liquidity/geopolitics) prevents overfitting to a single story and captures regime.
  - How: Company news requires ticker or full company name in headline; market news applies curated macro keywords (case-insensitive) and drops items with few related symbols to avoid single-stock noise. Returns only headline, summary, source for tight prompts.

- Tools facade: `src/tools/tools.py`
  - Why: A thin, stable boundary keeps UI/agents simple and makes failures predictable.
  - How: Helpers/clients raise on error; facade catches and returns `{ "error": "..." }`. Shapes are consistent, so prompts and agents don’t branch on edge cases.

### Agent mode (LangGraph ReAct)

- File: `src/agents/langgraph_agent.py`
- Description:
  - A LangGraph-based ReAct agent that binds the tools in `src/tools/tools.py` and routes requests to an LLM (OpenAI) with tool-use.
  - Useful for interactive research flows (ask questions, the agent decides which tools to call and synthesizes answers).
- Requirements:
  - `OPENAI_API_KEY` set in your environment.
- Example:

  ```python
  from src.agents.langgraph_agent import FinancialReActAgent

  agent = FinancialReActAgent(model_type="openai")  # uses OpenAI model
  result = agent.analyze_stock("AAPL")
  print(result["analysis"])  # Agent's synthesized answer after tool calls
  ```

### Forecasting and synthesis pipeline (FinGPT → RAG/OpenAI)

Why this design

- Separation of concerns: FinGPT focuses on short-horizon directionality; OpenAI focuses on cited explanation and decision framing.
- Better signal-to-noise: the forecaster receives only compact, high-signal inputs; long filings and verbose context are moved to the synthesis stage.
- Explainability: RAG over SEC filings surfaces risks/opportunities with citations; users see “why,” not just a number.

Stage 1: FinGPT forecaster (short-horizon view)

- Inputs (curated and compact):
  - Company basics (StockClient): ticker, company name, sector/industry, price, market cap, 52w range, 1d change.
  - Company headlines (NewsClient.company): 2–3 items (headline, summary, source) that explicitly match the ticker/name.
  - Macro headlines (NewsClient.market): 2–3 items (headline, summary, source) filtered by macro keywords and screened for single-stock noise.
  - Technical snapshot (TechnicalAnalyzer): minimal indicators and a synthesized verdict+confidence.
- Prompting constraints:
  - Keep sections concise (one-liners where possible).
  - Avoid pasting filings or long text; summarize instead.
  - Freeze date windows (from/to) for determinism when needed.
- Output: direction (up/down/flat) for next short window, rationale snippets, confidence.

Stage 2: OpenAI synthesis with RAG (explainability and decision)

- Retrieval:
  - Build/maintain FAISS index over SEC filings (10-K/10-Q/8-K) using sentence-transformers.
  - Retrieve top-k chunks for themes: guidance, liquidity, leverage, litigation, outlook.
  - Extract 3–5 bullets each for positives and concerns, with citations (filing + section/page).
- Synthesis prompt (to OpenAI):
  - Provide: (a) FinGPT forecast output, (b) condensed inputs (one-liners for basics/news/technicals), and (c) RAG bullets with citations.
  - Ask: 3–4 positives and 3–4 concerns; a probability-weighted next-week view; Buy/Hold/Sell with rationale and explicit risks; include citations.
- Output: a user-facing, cited recommendation and brief rationale aligned to the FinGPT direction (or highlighting divergence).

Optional scoring layer (aggregator)

- Combine multiple signals into a single score for thresholding:
  - Forecaster confidence (signed by direction)
  - Technical overall_signal × confidence
  - News breadth/recency (e.g., number of macro/company hits in the window)
  - Filing risk flags (penalize material risk bullets)
- Map thresholds to Buy/Hold/Sell; the OpenAI summary provides the narrative and citations.

Operational notes

- Cache upstream responses and set explicit date windows for reproducibility.
- Keep forecaster prompts small; rely on synthesis for depth.
- Handle errors at the facade; show graceful fallbacks in the UI.

### Minimal usage

```python
from src.tools.tools import get_stock_data, analyze_technical_indicators, get_news

print(get_stock_data("AAPL"))
print(analyze_technical_indicators("AAPL"))
print(get_news("AAPL", days_back=7))
```

### Error handling

- Helper layers (stock/news/technical) raise exceptions on failure
- Facade (`src/tools/tools.py`) catches and returns consistent `{"error": "..."}` dicts

### File structure (trimmed)

```
ai-investing-tool2/
├── README.md
├── requirements.txt
├── frontend/
│   └── streamlit_app.py
├── src/
│   ├── models/
│   │   └── fingpt_client.py
│   ├── tools/
│   │   ├── stock_data.py
│   │   ├── technical.py
│   │   ├── news.py
│   │   └── tools.py
│   ├── agents/
│   │   ├── langgraph_agent.py
│   │   └── react_agent.py
│   └── workflow/
│       ├── orchestrator.py (WIP)
│       └── scoring.py (WIP)
├── tests/
│   ├── test_tools.py
│   ├── test_news_sentiment.py
│   └── test_technical.py
└── data/
    ├── documents/
    ├── indices/
    └── cache/
```

### Notes

- Market news filtering uses a curated macro keyword list (case-insensitive) and drops single-company items by default. Pass `keywords=[]` to disable filtering, or provide a custom list.
- RAG over SEC filings is in progress; SEC retrieval + FAISS indexing + sentence-transformers will feed cited context into the final synthesis step.
