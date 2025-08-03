from src.config import Config
from src.data.sec_filings import SECFilingsFetcher
from src.processing.document_processor import DocumentProcessor
import json

def test_document_processing():
    """Test the document processing and chunking functionality"""
    config = Config()
    
    # Initialize components
    filings_fetcher = SECFilingsFetcher(config)
    doc_processor = DocumentProcessor(config)
    
    # Test with a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Testing Document Processing for {ticker}")
        print(f"{'='*60}")
        
        # Step 1: Fetch 10-K filing
        filing_data = filings_fetcher.get_latest_10k(ticker)
        
        if filing_data is None:
            print(f"Failed to fetch filing for {ticker}")
            continue
        
        print(f"✓ Fetched filing: {filing_data['file_size']} characters")
        
        # Step 2: Chunk the document
        chunks = doc_processor.chunk_document(filing_data['text'])
        
        print(f"✓ Created {len(chunks)} chunks")
        print(f"  - Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
        print(f"  - First chunk preview: {chunks[0][:100]}...")
        
        # Step 3: Create embeddings
        embeddings = doc_processor.create_embeddings(chunks)
        
        print(f"✓ Created embeddings: {embeddings.shape}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        
        # Step 4: Save everything
        doc_processor.save_chunks_and_embeddings(ticker, chunks, embeddings, filing_data)
        
        # Step 5: Test loading
        loaded_chunks, loaded_embeddings = doc_processor.load_chunks_and_embeddings(ticker)
        
        if loaded_chunks and loaded_embeddings is not None:
            print(f"✓ Successfully loaded saved data")
            print(f"  - Loaded chunks: {len(loaded_chunks)}")
            print(f"  - Loaded embeddings: {loaded_embeddings.shape}")
        else:
            print(f"✗ Failed to load saved data")
        
        print()

def show_metadata():
    """Show the metadata file to see what's been processed"""
    config = Config()
    
    try:
        with open(config.METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n{'='*60}")
        print("PROCESSED FILES METADATA")
        print(f"{'='*60}")
        
        for ticker, data in metadata.items():
            print(f"{ticker}:")
            print(f"  - Chunks: {data.get('chunks_count', 0)}")
            print(f"  - Has embeddings: {data.get('has_embeddings', False)}")
            print(f"  - Last updated: {data.get('last_updated', 'Unknown')}")
            print()
            
    except FileNotFoundError:
        print("No metadata file found. Run the test first.")

if __name__ == "__main__":
    test_document_processing()
    show_metadata() 