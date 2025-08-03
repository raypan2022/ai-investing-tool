import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (optional for now - we'll use yfinance which doesn't need keys)
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    # File Paths
    DATA_DIR = "data"
    FILINGS_DIR = os.path.join(DATA_DIR, "filings")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
    INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
    METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
    
    # Stock data configuration
    DEFAULT_LOOKBACK_DAYS = 30
    
    # Document processing configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # RAG configuration
    TOP_K_RETRIEVAL = 5 