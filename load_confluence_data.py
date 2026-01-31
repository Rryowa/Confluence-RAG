import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader

# Load environment variables from .env file
load_dotenv()

# --- Load credentials from environment variables ---
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# Existing imports and variable definitions remain the same

import os
import sys
import pickle
import time
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader

def clean_confluence_content(text):
    """
    Cleans Confluence content to reduce token usage.
    1. Removes image links.
    2. Unwraps HTML expanders (<details>, <summary>) and some tags.
    3. Simplifies links ([Text](path) -> Text).
    """
    if not text:
        return ""

    # 1. Remove Image Links: ![alt](path)
    # This removes the entire image reference.
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 2. Unwrap HTML Expanders (<details>, <summary>)
    # We replace <br/> with newlines first to preserve formatting
    text = re.sub(r'<br\s*/?>', '\n', text)
    # Remove the specific HTML tags but keep their inner content
    text = re.sub(r'</?(details|summary|p|ul|li)>', '', text)
    
    # 3. Simplify Links: [Text](path) -> Text
    # We capture the text group and replace the whole link with just the text
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)
    
    return text

load_dotenv()

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

CACHE_FILE = "confluence_docs.pkl"
CACHE_DURATION_HOURS = 24  # Cache is valid for 24 hours

def fetch_and_cache_documents():
    """Fetches documents from Confluence and saves them to a cache file."""
    # ... (rest of the function is the same as before)
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
        print("Error: Confluence environment variables are not set.")
        return []

    space_keys = ["VS2", "SUP"]
    all_documents = []

    for space_key in space_keys:
        try:
            print(f"\nFetching documents from space: {space_key}...")
            loader = ConfluenceLoader(
                url=CONFLUENCE_URL,
                username=CONFLUENCE_USERNAME,
                api_key=CONFLUENCE_API_TOKEN,
                space_key=space_key,
                include_attachments=False
            )
            documents = loader.load()
            
            # Clean content
            for doc in documents:
                doc.page_content = clean_confluence_content(doc.page_content)

            all_documents.extend(documents)
            print(f"Successfully fetched {len(documents)} documents from space: {space_key}")
        except Exception as e:
            print(f"Error fetching from space {space_key}: {e}")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_documents, f)
    print(f"\nSaved {len(all_documents)} documents to cache file: {CACHE_FILE}")
    return all_documents

def is_cache_valid():
    """Checks if the cache file exists and is within the valid duration."""
    if not os.path.exists(CACHE_FILE):
        return False
    cache_age_seconds = time.time() - os.path.getmtime(CACHE_FILE)
    if cache_age_seconds > CACHE_DURATION_HOURS * 3600:
        print(f"Cache is older than {CACHE_DURATION_HOURS} hours. Invalidating.")
        return False
    return True

def load_confluence_documents(force_refresh=False):
    """
    Loads documents from cache or fetches from Confluence based on cache validity
    or force_refresh flag.
    """
    if not force_refresh and is_cache_valid():
        print(f"Loading documents from valid cache: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} documents from cache.")
        return documents
    else:
        if force_refresh:
            print("Force refresh requested. Fetching from Confluence...")
        return fetch_and_cache_documents()

def main():
    """Main function to handle script execution and arguments."""
    force_refresh = "--force-refresh" in sys.argv
    documents = load_confluence_documents(force_refresh=force_refresh)
    if documents:
        print("\n--- First document from source ---")
        print(f"Source: {documents[0].metadata.get('source')}")
        print(f"Title: {documents[0].metadata.get('title')}")
        print(f"Content snippet: {documents[0].page_content[:500]}...")

if __name__ == "__main__":
    main()

