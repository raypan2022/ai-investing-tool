## AI Investing Tool — Data-Backed Recommendations and Research Assistant

### Overview

This project builds a stock analysis platform that delivers data-backed recommendations and a research chat assistant. It combines:

- Short-horizon forecasting (FinGPT forecaster service)
- Real-time stock basics (yfinance)
- Technical signals (pandas-ta)
- Financial news (Finnhub)
- Retrieval over SEC filings (RAG with FAISS + sentence-transformers)

The recommendation workflow composes these tools into a concise prompt for FinGPT, and optionally performs a second-stage synthesis using RAG citations.

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
  - Functional API: `fetch_stock_info`, `fetch_stock_history`
  - Client: `StockClient.get_info/get_history` (normalized fields: price, market cap, ratios, 52w range, etc.)
- Technical analysis: `src/tools/technical.py`
  - `TechnicalAnalyzer.analyze_stock` returns RSI/MACD/MAs, volatility, volume, levels, and a simple verdict+confidence
- News: `src/tools/news.py`
  - Functional API: `fetch_company_news`, `fetch_market_news`, `get_company_and_market_news`
  - Client: `NewsClient.company/market`
  - Market news filters: default macro keywords (case-insensitive) + drop single-stock items (few related symbols)
- Tools facade: `src/tools/tools.py`
  - `get_stock_data`, `analyze_technical_indicators`, `get_news`
  - Unifies error handling (helpers raise; facade catches and returns {'error': ...})

### FinGPT forecaster API

The forecaster is hosted behind a FastAPI endpoint. Example POST (prompt omitted for brevity):

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "... composed from stock basics, news (headline+summary+source), and technicals ..."
  }'
```

### Setup

1. Create env and install deps

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment
   Create `.env` at project root:

```
FINNHUB_API_KEY=your_finnhub_key
OPENAI_API_KEY=your_openai_key
FINGPT_ENDPOINT_URL=http://localhost:8000
FINGPT_API_KEY=optional_if_required
```

### Quick usage

- Tools facade

```python
from src.tools.tools import get_stock_data, analyze_technical_indicators, get_news

print(get_stock_data("AAPL"))
print(analyze_technical_indicators("AAPL"))
print(get_news("AAPL", days_back=7))
```

- Clients

```python
from src.tools.stock_data import StockClient
from src.tools.news import NewsClient

sc = StockClient()
info = sc.get_info("AAPL")
hist = sc.get_history("AAPL", period="6mo")

nc = NewsClient()
company = nc.company("AAPL", days_back=7, limit=3)
market = nc.market(days_back=7, limit=3)  # default macro keywords on headlines
```

### Tests

Run with printed output:

```bash
python -m pytest -s
```

- Integration prints:
  - `tests/test_tools.py`: prints stock snapshot, technical summary, and news (company+market)
  - `tests/test_news_sentiment.py`: prints a couple of company/market news lines (requires FINNHUB_API_KEY)

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
