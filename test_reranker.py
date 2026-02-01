import os
import torch
from transformers import AutoModel

RERANKER_MODEL_PATH = "./embeddings/jina-reranker-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading AutoModel from {RERANKER_MODEL_PATH} on {device}...")
try:
    model = AutoModel.from_pretrained(
        RERANKER_MODEL_PATH, 
        trust_remote_code=True, 
        tie_word_embeddings=False, 
        device_map=device
    )
    
    query = "What is the error?"
    docs = ["The error 505 is a server error.", "Banana is a fruit."]

    print("Testing .rerank() with minimal args...")
    results = model.rerank(query, docs)
    print(f"Results type: {type(results)}")
    print(f"Results: {results}")

except Exception as e:
    print(f"Error: {e}")
