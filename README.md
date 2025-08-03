# AI Investing Tool - Complete RAG-Based Stock Analysis

A comprehensive stock analysis system that combines technical analysis, fundamental data, economic context, and news sentiment to provide both short-term and long-term investment recommendations.

## 🚀 Features

### **Multi-Tiered Analysis**

- **Technical Analysis**: RSI, MACD, Bollinger Bands, Z-Score, and advanced statistical indicators
- **Fundamental Analysis**: PE ratios, market cap, price momentum, and company metrics
- **Economic Context**: Fed policy, market volatility, inflation trends, and economic sentiment
- **News & Filings**: SEC filings analysis, market sentiment, and key themes extraction

### **Dual Timeframe Recommendations**

- **Short-term (1-2 weeks)**: Technical-focused with entry timing
- **Long-term (6-12 months)**: Fundamental-focused with strategic positioning

### **ReAct Agent System**

- **Intelligent Reasoning**: ReAct-based agent that reasons about user intent
- **Tool Selection**: Automatically selects appropriate analysis tools
- **Specialized RAG Tools**: Separate tools for company news, economic data, and SEC filings
- **Chat Interface**: Interactive chat after deep research analysis

### **Professional-Grade Output**

- Algorithm-driven scoring with confidence levels
- Evidence-backed recommendations
- LLM-ready prompts for natural language analysis
- Comprehensive risk assessment

## 📁 Project Structure

```
ai-investing-tool/
├── src/
│   ├── agent/                    # ReAct agent system
│   │   ├── investment_agent.py   # Main investment agent
│   │   ├── react_agent.py        # ReAct reasoning engine
│   │   ├── rag_tools.py          # Specialized RAG tools
│   │   └── data_tools.py         # Data access tools
│   ├── analysis/                 # Analysis engines
│   │   ├── technical_analysis.py # Technical indicators & advanced statistics
│   │   ├── recommendation_engine.py # Multi-factor recommendation system
│   │   └── llm_analyzer.py       # Professional report generation
│   ├── data/                     # Data sources
│   │   ├── stock_data.py         # Stock price & fundamental data
│   │   ├── sec_filings.py        # SEC filings processing
│   │   └── economic_news.py      # Economic indicators & political events
│   ├── rag/                      # RAG system
│   │   └── vector_store.py       # FAISS vector search
│   ├── processing/               # Document processing
│   │   └── document_processor.py # Document chunking and embedding
│   └── tests/                    # Comprehensive test suite
├── data/                         # Data storage
└── requirements.txt              # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-investing-tool
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### **Complete Analysis Workflow**

Run the full analysis pipeline:

```bash
python -m src.tests.test_complete_analysis
```

This demonstrates:

- Stock data fetching (AAPL example)
- SEC filings processing
- Technical analysis with advanced statistics
- Economic context analysis
- Multi-factor recommendation generation
- Professional report formatting

### **ReAct Agent System**

Test the ReAct agent with chat capabilities:

```bash
python -m src.tests.test_react_agent
```

### **Individual Component Tests**

```bash
# Test technical analysis
python -m src.tests.test_technical_analysis

# Test recommendation engine
python -m src.tests.test_recommendation_engine

# Test stock data fetching
python -m src.tests.test_stock_data

# Test vector search
python -m src.tests.test_vector_search
```

## 📊 Analysis Components

### **1. Technical Analysis (`TechnicalAnalyzer`)**

- **Basic Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Advanced Statistics**: Z-Score, Rolling Statistics, Autocorrelation
- **Risk Metrics**: VaR, Sharpe Ratio, Max Drawdown
- **Statistical Tests**: ADF Test, GARCH Models, Change Point Detection

### **2. Recommendation Engine (`RecommendationEngine`)**

- **Short-term Weights**: Technical (50%), Fundamental (20%), Economic (15%), News (15%)
- **Long-term Weights**: Fundamental (50%), News (20%), Technical (20%), Economic (10%)
- **Confidence Levels**: High, Moderate, Low based on score deviation
- **Factor Scoring**: 0-100 scale for each analysis category

### **3. Economic Context (`EconomicNewsFetcher`)**

- **Economic Indicators**: Fed rates, inflation, unemployment, GDP
- **Market Conditions**: VIX, treasury yields, volatility regimes
- **Political Events**: Elections, policy changes, geopolitical events
- **Sentiment Analysis**: Fear & Greed Index, market sentiment

### **4. RAG System (`FAISSVectorStore`)**

- **Document Processing**: SEC filings, news articles, research reports
- **Vector Search**: Semantic similarity using FAISS
- **Relevance Scoring**: Document ranking and filtering
- **Theme Extraction**: Key topics and sentiment analysis

### **5. ReAct Agent System**

- **Specialized RAG Tools**:
  - `CompanyNewsRAGTool`: Company news and press releases
  - `EconomicDataRAGTool`: Economic indicators and market data
  - `SECFilingsRAGTool`: SEC filings and financial documents
- **Data Review Tools**:
  - `review_stock_data`: Stock fundamentals and market metrics
  - `review_technical_analysis`: Technical indicators and signals
- **ReAct Reasoning**: Think-Act-Observe-Respond cycle for tool selection

## 🎯 Recommendation Output

### **Short-term Analysis (1-2 weeks)**

```
📈 SHORT-TERM OUTLOOK (1-2 weeks):
   Recommendation: BUY
   Confidence: High
   Score: 78.0/100
   Factor Scores:
     • Technical: 82.5/100
     • Fundamental: 45.0/100
     • Economic: 90.0/100
     • News Sentiment: 95.0/100
```

### **Long-term Analysis (6-12 months)**

```
📊 LONG-TERM OUTLOOK (6-12 months):
   Recommendation: HOLD
   Confidence: Moderate
   Score: 67.0/100
   Factor Scores:
     • Fundamental: 45.0/100
     • News Sentiment: 95.0/100
     • Technical: 82.5/100
     • Economic: 90.0/100
```

## 🤖 ReAct Agent Chat Examples

### **User: "What's the current stock price?"**

```
ReAct Agent: "I need to get current stock data and fundamentals."
Tool Used: review_stock_data
Response: "Apple's current stock price is $202.38..."
```

### **User: "What are the RSI and MACD indicators showing?"**

```
ReAct Agent: "I need to get technical analysis data."
Tool Used: review_technical_analysis
Response: "The RSI is at 35.5 (oversold), MACD shows bearish signal..."
```

### **User: "What's the latest news about Apple?"**

```
ReAct Agent: "I need to search for recent company news."
Tool Used: rag_company_news
Response: "Recent news shows Apple announced AI investments..."
```

## ⚙️ Configuration

Key configuration parameters in `src/config.py`:

```python
# Technical Analysis
TECHNICAL_ANALYSIS_CONFIG = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# Recommendation Engine
RECOMMENDATION_CONFIG = {
    'high_confidence_threshold': 20,
    'moderate_confidence_threshold': 10,
    'buy_threshold': 70,
    'sell_threshold': 30
}
```

## 📈 Example Analysis Results

### **AAPL Analysis Summary**

- **Current Price**: $202.38
- **Market Cap**: $3.0T
- **PE Ratio**: 30.66
- **Technical Signal**: SELL (61.6% confidence)
- **Short-term Recommendation**: BUY (High confidence)
- **Long-term Recommendation**: HOLD (Moderate confidence)

## 🛠️ Technologies Used

**Core Analysis**

- `yfinance`: Real-time stock data and fundamentals
- `pandas-ta`: Technical analysis indicators
- `scipy`: Statistical analysis and significance testing
- `statsmodels`: Time series analysis and econometrics

**RAG & AI**

- `sentence-transformers`: Document embeddings using all-MiniLM-L6-v2
- `faiss-cpu`: High-performance vector similarity search
- `nltk`: Natural language processing and text analysis

**Advanced Statistics**

- `arch`: GARCH models for volatility forecasting
- `scikit-learn`: Machine learning utilities and preprocessing
- `numpy`: Numerical computations and array operations
- `pandas`: Data manipulation and analysis

## 🧪 Testing

Run the complete workflow:

```bash
python -m src.tests.test_complete_analysis
```

Test individual components:

```bash
python -m src.tests.test_technical_analysis
python -m src.tests.test_recommendation_engine
python -m src.tests.test_react_agent
```

## 🎯 What Makes This Different

**Comprehensive Analysis**: Combines technical, fundamental, economic, and news analysis in a single system.

**ReAct Agent Architecture**: Uses reasoning and acting to intelligently select tools and provide contextual responses.

**Specialized RAG Tools**: Separate optimized tools for different document types (news, economics, filings).

**Advanced Statistics**: Uses statistical methods (Z-scores, GARCH models, significance testing) for rigorous analysis.

**Professional Output**: Generates institutional-quality reports with specific recommendations and confidence levels.

**Dual Timeframe Analysis**: Provides both short-term trading and long-term investment perspectives.

## 📄 License

MIT License - see LICENSE file for details.
