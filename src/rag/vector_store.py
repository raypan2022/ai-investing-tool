import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.chunks = []
        self.metadata = {}
        self.embedding_model = None
        
    def create_index(self, embeddings: np.ndarray, chunks: List[str], 
                    metadata: Dict[str, Any], ticker: str):
        """Create and save FAISS index"""
        print(f"Creating FAISS index for {ticker}...")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = metadata
        
        # Save index and data
        self._save_index(ticker)
        
        print(f"✓ Created FAISS index with {len(chunks)} documents")
    
    def load_index(self, ticker: str) -> bool:
        """Load existing index for a ticker"""
        try:
            index_file = os.path.join(self.config.INDEXES_DIR, f"{ticker}_index.faiss")
            chunks_file = os.path.join(self.config.FILINGS_DIR, f"{ticker}_chunks.json")
            
            if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load chunks and metadata
            with open(chunks_file, 'r') as f:
                data = json.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            
            print(f"✓ Loaded FAISS index for {ticker} with {len(self.chunks)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading index for {ticker}: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Search for similar chunks"""
        if self.index is None:
            print("No index loaded. Please load an index first.")
            return []
        
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        # Create embedding for query
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        query_embedding = self.embedding_model.encode([query])
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_with_metadata(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search and return results with metadata"""
        results = self.search(query, top_k)
        
        enhanced_results = []
        for i, (chunk, score) in enumerate(results):
            enhanced_results.append({
                'rank': i + 1,
                'chunk': chunk,
                'score': score,
                'ticker': self.metadata.get('ticker', 'Unknown'),
                'filing_date': self.metadata.get('filing_date', 'Unknown'),
                'chunk_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
        
        return enhanced_results
    
    def _save_index(self, ticker: str):
        """Save FAISS index to disk"""
        os.makedirs(self.config.INDEXES_DIR, exist_ok=True)
        
        index_file = os.path.join(self.config.INDEXES_DIR, f"{ticker}_index.faiss")
        faiss.write_index(self.index, index_file)
        
        print(f"✓ Saved FAISS index to {index_file}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {'error': 'No index loaded'}
        
        return {
            'num_documents': len(self.chunks),
            'index_size': self.index.ntotal,
            'dimension': self.index.d,
            'ticker': self.metadata.get('ticker', 'Unknown'),
            'filing_date': self.metadata.get('filing_date', 'Unknown')
        } 