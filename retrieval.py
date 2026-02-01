import os
import pickle
import torch
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Load environment variables
load_dotenv()

# --- Configuration ---
# Models
EMBEDDING_MODEL_PATH = "./embeddings/qwen3-embedding-0.6b"
RERANKER_MODEL_PATH = "./embeddings/jina-reranker-v3"

# Data Stores
CHROMA_DB_PATH = "./chroma_db_v4"
PARENT_DOC_STORE_PATH = "parent_docs_store.pkl"
BM25_INDEX_PATH = "bm25_index.pkl"
BM25_CHUNKS_PATH = "bm25_chunks.pkl"

from langchain_core.embeddings import Embeddings

class QwenEmbeddings(Embeddings):
    def __init__(self, model_path: str, device: str):
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        # Qwen embedding usually works with raw query for symmetric tasks, 
        # or prepended with instruction.
        # Check config if instruction is needed. For now raw.
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.tolist()

class HybridRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing HybridRetriever on {self.device}...")

        # 1. Load Embedding Model
        print(f"Loading Embedding Model: {EMBEDDING_MODEL_PATH}...")
        self.embedding_model = QwenEmbeddings(EMBEDDING_MODEL_PATH, device=self.device)

        # 2. Load Vector Store (ChromaDB)
        print("Loading Vector Store...")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=self.embedding_model,
            collection_name="confluence_chunks"
        )

        # 3. Load BM25 Index
        print("Loading BM25 Index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(BM25_CHUNKS_PATH, "rb") as f:
            self.bm25_chunks = pickle.load(f)

        # 4. Load Parent Document Store
        print("Loading Parent Document Store...")
        with open(PARENT_DOC_STORE_PATH, "rb") as f:
            self.parent_store = pickle.load(f)

        # 5. Load Reranker Model (Jina V3)
        print(f"Loading Reranker: {RERANKER_MODEL_PATH}...")
        self.reranker_model = AutoModel.from_pretrained(
            RERANKER_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            tie_word_embeddings=False, # Fix for Identity head crash
            device_map=self.device
        )
        # Note: JinaForRanking does not require tokenizer separate load if using .rerank() 
        # but we might need it if we did manual. .rerank() handles it?
        # Actually .rerank() usually requires the model to have access to a tokenizer?
        # JinaForRanking implementation usually wraps the tokenizer or expects input_ids.
        # Wait, in test_reranker.py I did NOT pass a tokenizer to .rerank(), and I did not load one explicitly into the model.
        # But I DID NOT load a tokenizer in test_reranker.py at all!
        # And it worked!
        # This implies JinaForRanking loads its own tokenizer internally or defaults to something?
        # Checking test_reranker.py...
        # I imported AutoModel. loaded model. called model.rerank.
        # If model.rerank works without me providing a tokenizer, it must be self-contained.
        
        print("HybridRetriever initialized successfully.")

    def _rrf_merge(self, list1: List[Document], list2: List[Document], k=60) -> List[Document]:
        """
        Merges two lists of documents using Reciprocal Rank Fusion (RRF).
        """
        scores = {}
        
        # Helper to process a list
        def process_list(doc_list):
            for rank, doc in enumerate(doc_list):
                # Use chunk_id as unique key
                doc_id = doc.metadata.get("chunk_id")
                if not doc_id:
                    # Fallback if chunk_id missing (shouldn't happen)
                    doc_id = doc.page_content[:50]
                
                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0.0}
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        process_list(list1)
        process_list(list2)

        # Sort by accumulated score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def retrieve_candidates(self, query: str, top_k: int = 25) -> List[Document]:
        """
        Retrieves candidates using Vector Search + Keyword Search (BM25) and merges them.
        """
        print("  - Step 1A: Vector Search...")
        # A. Vector Search
        vector_docs = self.vectorstore.similarity_search(query, k=top_k)
        print(f"    -> Found {len(vector_docs)} vector results.")
        
        print("  - Step 1B: BM25 Search...")
        # B. Keyword Search (BM25)
        tokenized_query = query.split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.bm25_chunks, n=top_k)
        print(f"    -> Found {len(bm25_docs)} BM25 results.")

        # C. Merge (RRF)
        merged_docs = self._rrf_merge(vector_docs, bm25_docs)
        print(f"  - Step 1C: Merged into {len(merged_docs)} unique candidates.")
        return merged_docs

    def rerank_candidates(self, query: str, candidates: List[Document], top_n: int = 5) -> List[Document]:
        """
        Reranks a list of candidate documents using the Jina Reranker (.rerank method) with batching.
        """
        if not candidates:
            return []
            
        print(f"  - Step 2: Reranking {len(candidates)} candidates...")
        
        doc_texts = [doc.page_content for doc in candidates]
        
        # Configuration
        BATCH_SIZE = 4 # Conservative batch size for 0.6B-1.5B model on consumer GPU
        MAX_DOC_LENGTH = 512 # Match chunk size to avoid wasted padding
        
        all_results = []
        
        try:
            # Manual Batching Loop
            for i in range(0, len(doc_texts), BATCH_SIZE):
                batch_docs = doc_texts[i : i + BATCH_SIZE]
                
                # Call Jina's .rerank for this batch
                batch_results = self.reranker_model.rerank(
                    query=query, 
                    documents=batch_docs, 
                    max_doc_length=MAX_DOC_LENGTH,
                    top_n=None # Return all for this batch, sort later
                )
                
                # Adjust indices to match global list
                for res in batch_results:
                    res['index'] += i 
                    all_results.append(res)
                    
        except Exception as e:
            print(f"Rerank failed: {e}")
            return candidates[:top_n] # Fallback to RRF order

        # Map scores back to documents
        final_docs = []
        for res in all_results:
            idx = res['index']
            # Safety check
            if idx < len(candidates):
                doc = candidates[idx]
                doc.metadata["rerank_score"] = float(res['relevance_score'])
                final_docs.append(doc)
        
        # Sort by rerank score descending
        final_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
            
        if final_docs:
             print(f"    -> Top score: {final_docs[0].metadata['rerank_score']:.4f}")

        return final_docs[:top_n]

    def get_parent_document(self, chunk: Document) -> Document:
        """
        Retrieves the full parent document for a given chunk.
        """
        parent_id = chunk.metadata.get("parent_id")
        if parent_id and parent_id in self.parent_store:
            return self.parent_store[parent_id]
        return None

    def search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        End-to-end search: Retrieve -> Rerank -> Fetch Parent.
        Returns a list of dicts with 'chunk', 'parent', 'score'.
        """
        print(f"Searching for: '{query}'")
        
        # 1. Retrieve Candidates (Fast, Broad)
        candidates = self.retrieve_candidates(query, top_k=25) 
        
        # 2. Rerank (Slow, Precise)
        reranked_chunks = self.rerank_candidates(query, candidates, top_n=top_n)
        
        # 3. Fetch Parents (Context Expansion)
        results = []
        for chunk in reranked_chunks:
            parent = self.get_parent_document(chunk)
            results.append({
                "chunk": chunk,
                "parent": parent,
                "score": chunk.metadata.get("rerank_score", 0.0)
            })
            
        return results

if __name__ == "__main__":
    # Test run
    retriever = HybridRetriever()
    results = retriever.search("How do I fix the 505 error?")
    
    print("\n--- Search Results ---")
    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Score: {res['score']:.4f}):")
        print(f"Source: {res['chunk'].metadata.get('source')}")
        print(f"Chunk Content: {res['chunk'].page_content[:200]}...")
        if res['parent']:
            print(f"Parent Title: {res['parent'].metadata.get('title')}")
