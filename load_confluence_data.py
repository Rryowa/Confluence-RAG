import os
import sys
import pickle
import time
import re
from dotenv import load_dotenv
from atlassian import Confluence
from langchain_core.documents import Document
from markdownify import markdownify as md

# Load environment variables from .env file
load_dotenv()

# --- Load credentials from environment variables ---
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

CACHE_FILE = "confluence_docs.pkl"
CACHE_DURATION_HOURS = 24  # Cache is valid for 24 hours

def custom_html_process(html_content):
    """
    Custom HTML to Markdown conversion using markdownify.
    - heading_style="ATX" -> Forces # instead of underlined headers
    - wrap=False -> Prevents 80-char line breaks
    - bullets="-" -> Standardizes bullet points
    """
    if not html_content:
        return ""
    
    return md(
        html_content, 
        heading_style="ATX", 
        wrap=False, 
        bullets="-"
    )

def extract_bold_keywords(text):
    """
    Extracts bold text to use as keywords.
    Strategy A: Metadata Injection.
    """
    if not text:
        return []
    # Find all bold text: **text**
    # Use non-greedy match, ensure some length
    matches = re.findall(r'\*\*([^*]+)\*\*', text)
    # Clean and filter
    keywords = [m.strip() for m in matches if len(m.strip()) > 3] # Filter very short noise
    return list(set(keywords)) # Deduplicate

def clean_confluence_content(text):
    """
    Cleans Confluence content.
    Input is already Markdown (converted via custom_html_process).
    1. Removes Hex color codes (e.g. #DEEBFF).
    2. Removes image links.
    3. Cleans URLs (Angle brackets, tracking params).
    4. Simplifies links.
    5. Flattens Tables.
    6. Removes Conversational Noise (Visual anchors, procedural, quiz).
    """
    if not text:
        return ""

    # --- Noise Reduction ---
    # 1. Visual Anchors
    text = re.sub(r'(?i)\(pic above\)', '', text)
    text = re.sub(r'(?i)\(see image below\)', '', text)
    text = re.sub(r'(?i)this is what you are going to see:?', '', text)
    
    # 2. Procedural Navigation
    text = re.sub(r'(?i)click on file\s*→\s*make a copy', '', text)
    text = re.sub(r'(?i)file\s*→\s*make a copy', '', text)
    
    # 3. Quiz Feedback
    text = re.sub(r'(?i)answer \d+ is the correct answer!?', '', text)
    text = re.sub(r'(?i)if you got this, congrats!?', '', text)
    
    # --- Standard Cleaning ---
    
    # 4. Remove Hex Codes (standalone or start of line, e.g. #DEEBFF)
    # Often Confluence puts these in macros or spans that markdownify outputs as text
    text = re.sub(r'(?m)^#([A-Fa-f0-9]{6})\s*$', '', text) # Standalone lines
    text = re.sub(r'#([A-Fa-f0-9]{6})\b', '', text) # Inline

    # 5. Remove Image Links: ![alt](path)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 6. Clean URLs
    # Remove angle brackets: <http...> -> http...
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    # Remove atlOrigin and other common noise params
    text = re.sub(r'\?atlOrigin=[a-zA-Z0-9%_\-]+', '', text)
    text = re.sub(r'&atlOrigin=[a-zA-Z0-9%_\-]+', '', text)
    
    # 7. Simplify Links: [Text](path) -> Text
    text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', text)

    # 8. Flatten Tables (Simple heuristic)
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|') and '---' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2: # Keep logic similar to original but slightly more robust
                # Join with colon if 2 parts, or just space/pipe for more?
                # Original logic: if len(parts) == 2: new_lines.append(f"{parts[0]}: {parts[1]}")
                # Let's stick to the original logic for now as requested by "Conventions"
                if len(parts) == 2:
                    new_lines.append(f"{parts[0]}: {parts[1]}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    text = '\n'.join(new_lines)
    
    return text

def fetch_and_cache_documents():
    """Fetches documents from Confluence and saves them to a cache file."""
    if not all([CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN]):
        print("Error: Confluence environment variables are not set.")
        return []

    # Initialize Atlassian Confluence API
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_API_TOKEN,
        cloud=True 
    )

    space_keys = ["VS2", "SUP"]
    all_documents = []

    for space_key in space_keys:
        try:
            print(f"\nFetching documents from space: {space_key}...")
            
            # Fetch all pages from the space
            # expand='body.storage,version' gets the raw XML/HTML and version info
            start = 0
            limit = 50
            pages = []
            
            while True:
                print(f"Fetching pages {start} to {start+limit}...")
                batch = confluence.get_all_pages_from_space(
                    space=space_key, 
                    start=start, 
                    limit=limit, 
                    expand='body.storage,version', 
                    content_type='page'
                )
                if not batch:
                    break
                pages.extend(batch)
                start += limit
                # Safety break/limit if needed, or rely on loop until empty
                if len(batch) < limit:
                    break

            print(f"Fetched {len(pages)} raw pages. Processing...")

            documents = []
            for page in pages:
                page_id = page.get("id")
                title = page.get("title")
                web_url = page.get("_links", {}).get("webui", "")
                
                # Extract Raw HTML
                raw_html = page.get("body", {}).get("storage", {}).get("value", "")
                
                # Apply Custom Pipeline (HTML -> Markdown)
                clean_text = custom_html_process(raw_html)
                
                # Apply Sanitization (Regex Rules)
                final_text = clean_confluence_content(clean_text)
                
                # Extract Keywords (Strategy A)
                keywords = extract_bold_keywords(final_text)
                
                # Metadata handling
                # Extract last_updated if available from version
                last_updated = page.get("version", {}).get("when", "")
                
                # Create Document Object
                doc = Document(
                    page_content=final_text,
                    metadata={
                        "source": f"{CONFLUENCE_URL.rstrip('/')}/wiki{web_url}",
                        "title": title,
                        "id": page_id,
                        "last_updated": last_updated,
                        "keywords": keywords
                    }
                )
                documents.append(doc)

            all_documents.extend(documents)
            print(f"Successfully processed {len(documents)} documents from space: {space_key}")
            
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
        print(f"Keywords: {documents[0].metadata.get('keywords', [])[:5]}...")
        print(f"Content snippet: {documents[0].page_content[:500]}...")

        if len(documents) > 4:
            print("\n--- 5th Document from source ---")
            print(f"Source: {documents[4].metadata.get('source')}")
            print(f"Title: {documents[4].metadata.get('title')}")
            print(f"Keywords: {documents[4].metadata.get('keywords', [])[:5]}...")
            print(f"Content snippet:\n{documents[4].page_content[:1000]}...")

if __name__ == "__main__":
    main()