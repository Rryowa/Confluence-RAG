import os
import torch
import shutil
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from typing import List

# Import llama_cpp for GGUF support
from llama_cpp import Llama

from load_confluence_data import load_confluence_documents

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Child chunks for vector search (small, focused)
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# Path to the GGUF model file
EMBEDDING_MODEL_NAME = "C:/Users/Rryo/Gemini/Confluence-RAG/embeddings/jina-embeddings-v4/jina-embeddings-v4.gguf"

CHROMA_DB_PATH = "./chroma_db_v4"
PARENT_DOC_STORE_PATH = "parent_docs_store.pkl"

class JinaEmbeddings(Embeddings):
    """
    A custom wrapper for Jina Embeddings v4 (GGUF) using llama-cpp-python.
    """
    def __init__(self, model_name_or_path: str, device: str = None):
        # device param is unused as llama-cpp handles gpu via n_gpu_layers
        print(f"Loading GGUF model from: {model_name_or_path}")
        self.model = Llama(
            model_path=model_name_or_path,
            embedding=True,
            n_gpu_layers=-1, # Use all available GPU layers
            verbose=True,    # Set to True to see loading logs
            n_ctx=2048       # Set context window suitable for embeddings
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            # Note: task instructions (retrieval.passage) are typically prepended to text 
            # for GGUF models if needed. Jina V4 GGUF might expect raw text.
            # We will pass raw text for now.
            response = self.model.create_embedding(text)
            embeddings.append(response['data'][0]['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.model.create_embedding(text)
        return response['data'][0]['embedding']

def create_and_store_embeddings():
    """
    Loads documents, uses ParentDocumentRetriever to split them into child chunks,
    stores chunks in ChromaDB, and full parent documents in InMemoryStore (pickled).
    """
    # 1. Load documents
    print("Loading Confluence documents...")
    documents = load_confluence_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 2. Setup Stores
    print(f"Setting up ChromaDB at {CHROMA_DB_PATH}...")
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Removing existing ChromaDB at {CHROMA_DB_PATH}...")
        shutil.rmtree(CHROMA_DB_PATH)
    
    # Load Embedding Model
    print(f"Loading Jina embeddings model: {EMBEDDING_MODEL_NAME}...")
    embeddings = JinaEmbeddings(model_name_or_path=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    # Storage for Parent Documents
    store = InMemoryStore()
    
    # 3. Setup ParentDocumentRetriever
    print(f"Configuring ParentDocumentRetriever with child_chunk_size={CHILD_CHUNK_SIZE}...")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        # parent_splitter=None, # Default: Documents are not split (Full Parent Mode)
    )

    # 4. Add Documents (Ingestion)
    print("Adding documents to retriever (this generates chunks and indexes them)...")
    retriever.add_documents(documents, ids=None)
    print("Documents added.")

    # 5. Persist Everything
    print("Persisting ChromaDB...")
    vectorstore.persist()
    
    print(f"Saving Parent Document Store to {PARENT_DOC_STORE_PATH}...")
    with open(PARENT_DOC_STORE_PATH, "wb") as f:
        pickle.dump(store, f)
        
    print("Ingestion complete. Parent-Child retrieval ready.")

if __name__ == "__main__":
    create_and_store_embeddings()

if __name__ == "__main__":
    create_and_store_embeddings()
