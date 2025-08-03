from src.config import Config
from src.processing.document_processor import DocumentProcessor
from src.rag.vector_store import FAISSVectorStore
import json

def test_vector_search():
    """Test the FAISS vector search functionality"""
    config = Config()
    
    # Initialize components
    doc_processor = DocumentProcessor(config)
    vector_store = FAISSVectorStore(config)
    
    # Test with AAPL (should already have embeddings from Component 2)
    ticker = 'AAPL'
    
    print(f"\n{'='*60}")
    print(f"Testing Vector Search for {ticker}")
    print(f"{'='*60}")
    
    # Step 1: Load existing chunks and embeddings
    chunks, embeddings = doc_processor.load_chunks_and_embeddings(ticker)
    
    if chunks is None or embeddings is None:
        print(f"No existing data found for {ticker}. Please run Component 2 first.")
        return
    
    print(f"✓ Loaded {len(chunks)} chunks and embeddings")
    
    # Step 2: Create FAISS index
    # Load metadata
    chunks_file = f"data/filings/{ticker}_chunks.json"
    with open(chunks_file, 'r') as f:
        data = json.load(f)
        metadata = data['metadata']
    
    vector_store.create_index(embeddings, chunks, metadata, ticker)
    
    # Step 3: Test various search queries
    test_queries = [
        "What is the company's business model?",
        "What are the main risk factors?",
        "How much revenue did the company generate?",
        "What is the company's strategy?",
        "What are the legal proceedings?",
        "What is the dividend policy?",
        "How does the company compete?",
        "What are the financial results?"
    ]
    
    print(f"\n{'='*60}")
    print("SEARCH RESULTS")
    print(f"{'='*60}")
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        results = vector_store.search_with_metadata(query, top_k=3)
        
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.3f})")
            print(f"Preview: {result['chunk_preview']}")
            print()
    
    # Step 4: Test index statistics
    print(f"\n{'='*60}")
    print("INDEX STATISTICS")
    print(f"{'='*60}")
    
    stats = vector_store.get_index_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Step 5: Test loading existing index
    print(f"\n{'='*60}")
    print("TESTING INDEX LOADING")
    print(f"{'='*60}")
    
    # Create new vector store instance
    new_vector_store = FAISSVectorStore(config)
    
    if new_vector_store.load_index(ticker):
        print("✓ Successfully loaded existing index")
        
        # Test search with loaded index
        test_query = "What is the company's revenue?"
        results = new_vector_store.search_with_metadata(test_query, top_k=2)
        
        print(f"\nTest search with loaded index:")
        print(f"Query: {test_query}")
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.3f})")
            print(f"Preview: {result['chunk_preview']}")
    else:
        print("✗ Failed to load existing index")

def test_similarity_search():
    """Test semantic similarity with different query variations"""
    config = Config()
    vector_store = FAISSVectorStore(config)
    
    # Load AAPL index
    if not vector_store.load_index('AAPL'):
        print("Please run the main test first to create the index.")
        return
    
    print(f"\n{'='*60}")
    print("SEMANTIC SIMILARITY TEST")
    print(f"{'='*60}")
    
    # Test similar queries that should return similar results
    similar_queries = [
        "What does the company do?",
        "What is the business description?",
        "How does the company operate?",
        "What is the company's main business?"
    ]
    
    for query in similar_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        results = vector_store.search_with_metadata(query, top_k=2)
        
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.3f})")
            print(f"Preview: {result['chunk_preview'][:100]}...")
        print()

if __name__ == "__main__":
    test_vector_search()
    test_similarity_search() 