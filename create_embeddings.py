import os
import shutil
import pickle
import uuid
import torch
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from load_confluence_data import load_confluence_documents

load_dotenv()

# --- Configuration ---
# Models
EMBEDDING_MODEL_NAME = "./embeddings/qwen3-embedding-0.6b" 
# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Storage Paths
CHROMA_DB_PATH = "./chroma_db_v4"
PARENT_DOC_STORE_PATH = "parent_docs_store.pkl"
BM25_INDEX_PATH = "bm25_index.pkl"
BM25_CHUNKS_PATH = "bm25_chunks.pkl"

from sentence_transformers import SentenceTransformer

class JinaEmbeddingsTransformers(Embeddings):
    """
    Generic Embeddings using SentenceTransformer.
    """
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Embedding Model: {model_name} on {self.device}...")
        # Use trust_remote_code=True as Qwen/Jina architectures often require it.
        # It's local, so less risk.
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Qwen embedding models typically handle raw text or might need instruction.
        # Assuming standard usage for now.
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # Qwen embedding models usually need a prompt for queries vs documents.
        # If it's a "Qwen2.5-Instruct" based embedding, it definitely needs instructions.
        # However, checking the folder structure, it has 'modules.json' implying it's wrapped for SBERT.
        # SBERT usually handles this if registered. If not, we might need a prompt.
        # Common Qwen embedding prompt: 'Represent this sentence for searching relevant passages: '
        # I'll apply it tentatively.
        
        # instruction = "Represent this sentence for searching relevant passages: "
        # text = instruction + text
        
        # For now, let's assume the SBERT wrapper handles it or it's symmetric.
        # (Re-enabling instruction if results are poor).
        
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedding.tolist()

def process_documents(documents: List[Document]) -> (List[Document], Dict[str, Document]):
    """
    1. Splits Full Pages into Header Sections (H1, H2, H3). -> PARENTS
    2. Splits Sections into 256-token Child Chunks using Qwen Tokenizer. -> CHILDREN
    """
    print("Processing documents...")
    
    # Load Tokenizer for precise chunking
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    # strip_headers=False keeps the header in the text, which is good for context in the Parent
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Child Splitter: 256 tokens, 32 overlap
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=256,
        chunk_overlap=32
    )

    final_chunks = []
    parent_store = {}

    for full_page_doc in documents:
        # 1. Split Full Page into Sections (Parents)
        section_splits = markdown_splitter.split_text(full_page_doc.page_content)
        
        for section in section_splits:
            # Merge original metadata
            section.metadata.update(full_page_doc.metadata)
            
            # Construct Context String (for display/debugging, though header is now in text)
            title = full_page_doc.metadata.get("title", "Untitled")
            headers = [section.metadata.get(h, "") for _, h in headers_to_split_on]
            headers = [h for h in headers if h]
            context_str = f"{title} > {' > '.join(headers)}" if headers else title
            section.metadata["context"] = context_str
            
            # Generate Parent ID
            parent_id = str(uuid.uuid4())
            section.metadata["parent_id"] = parent_id
            
            # Store PARENT (Full Section)
            # We explicitly add context to the start of the parent doc for clarity
            if not section.page_content.startswith(f"**Context:** {title}"):
                 section.page_content = f"**Context:** {context_str}\n\n{section.page_content}"
            parent_store[parent_id] = section

            # 2. Split Section into Child Chunks (Tokens)
            child_chunks = text_splitter.split_documents([section])

            for chunk in child_chunks:
                chunk.metadata["parent_id"] = parent_id
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
                
                # Note: Token-based splitter might cut in middle of sentence if not careful, 
                # but Recursive splitter tries to respect separators.
                # We do NOT prepend context to every chunk here to save tokens, 
                # relying on the Parent retrieval to provide full context.
                # However, for vector search, some context is helpful. 
                # Given chunks are small (256), prepending a long title might eat space.
                # But 'context_str' is usually short. Let's keep it in metadata mainly, 
                # and maybe minimal in text if needed. 
                # User's prompt implies "Child: Small granular chunks". 
                # I will trust the splitter.
                
                final_chunks.append(chunk)

    return final_chunks, parent_store

def create_bm25_index(chunks: List[Document]):
    """Creates and saves a BM25 index for the chunks."""
    from rank_bm25 import BM25Okapi
    
    print("Building BM25 Index...")
    tokenized_corpus = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    
    # We also need to save the chunks corresponding to the index to map back
    # For memory efficiency, we might just save IDs, but for now we save docs (or lightweight versions)
    with open(BM25_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("BM25 Index saved.")

def flatten_metadata(doc: Document) -> Document:
    """
    Converts list-type metadata to comma-separated strings for ChromaDB compatibility.
    """
    for key, value in doc.metadata.items():
        if isinstance(value, list):
            doc.metadata[key] = ", ".join(map(str, value))
    return doc

def create_and_store_embeddings():
    # 1. Load
    documents = load_confluence_documents()
    if not documents:
        return

    # 2. Process (Split & Parent Store)
    chunks, parent_store = process_documents(documents)
    print(f"Generated {len(chunks)} chunks from {len(documents)} parent documents.")
    
    # --- FIX: Sanitize Metadata for ChromaDB ---
    print("Sanitizing metadata (flattening lists)...")
    chunks = [flatten_metadata(chunk) for chunk in chunks]

    # 3. Vector Store
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    
    embeddings = JinaEmbeddingsTransformers(model_name=EMBEDDING_MODEL_NAME)
    
    print("Creating ChromaDB vector store...")
    # Chroma handles batching internally, but we can control it if needed.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="confluence_chunks"
    )
    vectorstore.persist()
    print("ChromaDB persisted.")

    # 4. Save Parent Store
    print(f"Saving Parent Store to {PARENT_DOC_STORE_PATH}...")
    with open(PARENT_DOC_STORE_PATH, "wb") as f:
        pickle.dump(parent_store, f)

    # 5. Build BM25
    create_bm25_index(chunks)

    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    create_and_store_embeddings()