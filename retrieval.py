import os
import pickle
import torch
import numpy as np
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

# Import JinaEmbeddings from our existing script
# Ensuring create_embeddings.py has the class available and doesn't run main logic on import
try:
    from create_embeddings import JinaEmbeddings, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, BM25_INDEX_PATH, BM25_CHUNKS_PATH
except ImportError:
    # Fallback if create_embeddings is not importable (e.g. strict script nature)
    # But since it is in the same dir, it should work.
    pass

# Reranker Configuration
RERANKER_MODEL_PATH = "./embeddings/jina-reranker-v3"

class HybridRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing HybridRetriever on {self.device}...")

        # 1. Load Vector Store (ChromaDB)
        print("Loading Vector Store...")
        self.embedding_model = JinaEmbeddings(
            model_name_or_path=EMBEDDING_MODEL_NAME, 
            device=self.device
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=self.embedding_model
        )

        # 2. Load BM25 Index
        print("Loading BM25 Index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(BM25_CHUNKS_PATH, "rb") as f:
            self.bm25_chunks = pickle.load(f)

        # 3. Load Reranker Model (Jina V3)
        print(f"Loading Reranker: {RERANKER_MODEL_PATH}...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH, 
            trust_remote_code=True
        )
        self.reranker_model = AutoModel.from_pretrained(
            RERANKER_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.reranker_model.eval()
        print("HybridRetriever initialized successfully.")

    def _rrf_merge(self, list1: List[Document], list2: List[Document], k=60) -> List[Document]:
        """
        Merges two lists of documents using Reciprocal Rank Fusion (RRF).
        """
        scores = {}
        
        # Helper to process a list
        def process_list(doc_list):
            for rank, doc in enumerate(doc_list):
                # Use page_content as a unique key for deduplication (simplification)
                # Ideally use a unique ID if available.
                doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("start_index", "")) + doc.page_content[:20]
                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0.0}
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        process_list(list1)
        process_list(list2)

        # Sort by accumulated score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def retrieve_candidates(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Retrieves candidates using Vector Search + Keyword Search (BM25) and merges them.
        """
        # A. Vector Search
        vector_docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # B. Keyword Search (BM25)
        tokenized_query = query.split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.bm25_chunks, n=top_k)

        # C. Merge (RRF)
        merged_docs = self._rrf_merge(vector_docs, bm25_docs)
        return merged_docs

    def rerank_candidates(self, query: str, candidates: List[Document], top_n: int = 5) -> List[Document]:
        """
        Reranks a list of candidate documents using the Jina Reranker.
        """
        if not candidates:
            return []

        # Prepare pairs for the model
        pairs = [[query, doc.page_content] for doc in candidates]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=1024
            ).to(self.device)
            
            outputs = self.reranker_model(**inputs)
            
            # The custom JinaForRanking model returns logits as the first element or .logits
            if hasattr(outputs, 'logits'):
                scores = outputs.logits
            else:
                scores = outputs[0]
                
            scores = scores.view(-1).float()
            
            # Sigmoid to get 0-1 scores (optional, but good for thresholding)
            scores = torch.sigmoid(scores)

        # Sort candidates by score
        scored_candidates = sorted(
            zip(candidates, scores.cpu().numpy()), 
            key=lambda x: x[1], 
            reverse=True
        )

        # Return top N
        return [doc for doc, score in scored_candidates[:top_n]]

    def search(self, query: str) -> List[Document]:
        """
        End-to-end search: Retrieve -> Rerank.
        """
        print(f"Searching for: '{query}'")
        candidates = self.retrieve_candidates(query, top_k=20)
        print(f"Retrieved {len(candidates)} candidates from Hybrid Search.")
        
        final_docs = self.rerank_candidates(query, candidates, top_n=5)
        print(f"Reranked to top {len(final_docs)} results.")
        return final_docs

if __name__ == "__main__":
    # Test run
    retriever = HybridRetriever()
    results = retriever.search("How do I fix the 505 error?")
    
    print("\n--- Search Results ---")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:200]}...")
