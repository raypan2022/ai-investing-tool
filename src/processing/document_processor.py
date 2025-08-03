import re
import json
import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.embedding_model = None  # Will be loaded when needed
        
    def chunk_document(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split document into overlapping chunks"""
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if overlap is None:
            overlap = self.config.CHUNK_OVERLAP
            
        print(f"Chunking document with size={chunk_size}, overlap={overlap}")
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        return text.strip()
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last part of text for overlap"""
        words = text.split()
        if len(words) <= overlap_size:
            return text
        return " ".join(words[-overlap_size:])
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if self.embedding_model is None:
            print("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def save_chunks_and_embeddings(self, ticker: str, chunks: List[str], 
                                 embeddings: np.ndarray, metadata: Dict[str, Any]):
        """Save chunks and embeddings to disk"""
        # Create directories if they don't exist
        os.makedirs(self.config.FILINGS_DIR, exist_ok=True)
        os.makedirs(self.config.EMBEDDINGS_DIR, exist_ok=True)
        
        # Save chunks
        chunks_file = os.path.join(self.config.FILINGS_DIR, f"{ticker}_chunks.json")
        with open(chunks_file, 'w') as f:
            json.dump({
                'chunks': chunks,
                'metadata': metadata
            }, f, indent=2)
        
        # Save embeddings
        embeddings_file = os.path.join(self.config.EMBEDDINGS_DIR, f"{ticker}_embeddings.npy")
        np.save(embeddings_file, embeddings)
        
        # Update metadata
        self._update_metadata(ticker, metadata, len(chunks))
        
        print(f"Saved {len(chunks)} chunks and embeddings for {ticker}")
    
    def _update_metadata(self, ticker: str, metadata: Dict[str, Any], chunks_count: int):
        """Update the global metadata file"""
        metadata_file = self.config.METADATA_FILE
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                global_metadata = json.load(f)
        else:
            global_metadata = {}
        
        global_metadata[ticker] = {
            **metadata,
            'last_updated': metadata.get('downloaded_at', ''),
            'chunks_count': chunks_count,
            'has_embeddings': True
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(global_metadata, f, indent=2)
    
    def load_chunks_and_embeddings(self, ticker: str) -> tuple:
        """Load existing chunks and embeddings for a ticker"""
        chunks_file = os.path.join(self.config.FILINGS_DIR, f"{ticker}_chunks.json")
        embeddings_file = os.path.join(self.config.EMBEDDINGS_DIR, f"{ticker}_embeddings.npy")
        
        if not (os.path.exists(chunks_file) and os.path.exists(embeddings_file)):
            return None, None
        
        # Load chunks
        with open(chunks_file, 'r') as f:
            data = json.load(f)
            chunks = data['chunks']
            metadata = data['metadata']
        
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        return chunks, embeddings 