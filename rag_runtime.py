import os
import json
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Provider Libraries
from openai import OpenAI  # Used for OpenRouter
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from retrieval import HybridRetriever

load_dotenv()

# --- Configuration ---
# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Models - Primary (Gemini)
GEMINI_FAST_MODEL = "gemini-3-flash-preview"
GEMINI_DEEP_MODEL = "gemini-3-pro-preview"

# Models - Failover (OpenRouter/DeepSeek)
OR_FAST_MODEL = "deepseek/deepseek-v3.2"           # No reasoning (Fast)
OR_DEEP_MODEL = "deepseek/deepseek-v3.2-speciale"  # Native reasoning (Deep)

class LLMWrapper:
    """Unified wrapper with automatic Gemini -> OpenRouter failover."""
    def __init__(self):
        self.gemini_client = None
        self.openrouter_client = None
        
        # Initialize Gemini
        if GEMINI_API_KEY and genai:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"Primary Client Initialized: Gemini ({GEMINI_FAST_MODEL} / {GEMINI_DEEP_MODEL})")
            except Exception as e:
                print(f"Failed to init Gemini: {e}")

        # Initialize OpenRouter (Failover)
        if OPENROUTER_API_KEY:
            try:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )
                print(f"Failover Client Initialized: OpenRouter ({OR_FAST_MODEL} / {OR_DEEP_MODEL})")
            except Exception as e:
                print(f"Failed to init OpenRouter: {e}")

        if not self.gemini_client and not self.openrouter_client:
            print("WARNING: No valid API keys found. Responses will be mocked.")

    def _call_gemini(self, model: str, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> str:
        """Helper to call Gemini API."""
        if not self.gemini_client:
            raise ImportError("Gemini client not initialized")
            
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
            system_instruction=system_prompt
        )
        
        response = self.gemini_client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=config
        )
        return response.text

    def _call_openrouter(self, model: str, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> str:
        """Helper to call OpenRouter API. No extra_body needed for these models."""
        if not self.openrouter_client:
            raise ImportError("OpenRouter client not initialized")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        response = self.openrouter_client.chat.completions.create(**params)
        return response.choices[0].message.content

    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False, mode: str = "FAST", temperature: float = 0.0) -> str:
        """
        Generates response trying Gemini first, then failing over to OpenRouter.
        mode: "FAST" or "DEEP" determines which model set to use.
        """
        # Determine Models based on Mode
        if mode == "DEEP":
            primary_model = GEMINI_DEEP_MODEL
            failover_model = OR_DEEP_MODEL
        else: # FAST
            primary_model = GEMINI_FAST_MODEL
            failover_model = OR_FAST_MODEL

        # --- Attempt 1: Gemini ---
        if self.gemini_client:
            try:
                return self._call_gemini(primary_model, system_prompt, user_prompt, temperature, json_mode)
            except Exception as e:
                # Check for rate limits (429) or resource exhaustion
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    print(f"\n[⚠️ RATE LIMIT] Gemini ({primary_model}) overloaded. Switching to OpenRouter...")
                else:
                    print(f"Gemini Error ({primary_model}): {e}. Switching to OpenRouter...")
        
        # --- Attempt 2: OpenRouter (Failover) ---
        if self.openrouter_client:
            try:
                print(f"[Failover] Calling OpenRouter ({failover_model})...")
                return self._call_openrouter(failover_model, system_prompt, user_prompt, temperature, json_mode)
            except Exception as e:
                print(f"OpenRouter Error ({failover_model}): {e}")
                return "ERROR_GENERATING_RESPONSE"

        return "MOCKED_RESPONSE"

class RAGRuntime:
    def __init__(self):
        print("Initializing RAG Runtime...")
        self.retriever = HybridRetriever()
        self.llm = LLMWrapper()

    def route_query(self, query: str) -> str:
        """Phase 3, Step 0: The Router."""
        print("Routing query...")
        prompt = f"""
        You are an expert query dispatcher.
        Classify the User Query into "FAST" or "DEEP".

        LOGIC:
        - FAST: Simple fact, policy lookup, "how-to".
        - DEEP: Strategy, comparison, reasoning, complex analysis.

        OUTPUT JSON:
        {{ "route": "FAST" | "DEEP" }}

        User Query: "{query}"
        """
        
        # Use FAST mode (Gemini Flash -> DeepSeek v3.2)
        response = self.llm.generate(
            system_prompt="You are a router. Output valid JSON.",
            user_prompt=prompt,
            json_mode=True,
            mode="FAST", 
            temperature=0.0
        )
        print(f"  Router raw response: {response}")
        
        if response == "MOCKED_RESPONSE":
            return "FAST"
            
        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            route = data.get("route", "FAST")
            print(f"  Router parsed route: {route}")
            return route
        except Exception as e:
            print(f"Router parsing error: {e}")
            return "FAST"

    def format_context(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks for the LLM."""
        context_str = ""
        for i, res in enumerate(retrieved_results):
            chunk = res['chunk']
            parent = res['parent']
            source_id = i + 1 
            
            context_str += f"\n--- [Source ID: {source_id}] ---"
            context_str += f"\nTitle: {parent.metadata.get('title') if parent else 'Unknown'}"
            context_str += f"\nContent:\n{chunk.page_content}"
        return context_str

    def run_fast_track(self, query: str):
        """Phase 3, Step 8: Fast Lookup."""
        print("--- Executing FAST Track ---")
        results = self.retriever.search(query, top_n=3)
        context = self.format_context(results)
        
        system_prompt = "You are a Confluence Knowledge Assistant. Answer using the context. Cite [Source ID]."
        user_prompt = f"Context:\n{context}\n\nUser Query: \"{query}\""
        
        # Use FAST mode (Gemini Flash -> DeepSeek v3.2)
        answer = self.llm.generate(
            system_prompt, 
            user_prompt,
            mode="FAST",
            temperature=0.0
        )
        return answer, results

    def run_deep_track(self, query: str):
        """Phase 3, Step 9: The "Check & Drill" Protocol."""
        print("--- Executing DEEP Track ---")
        
        # Step 1: Broad Sweep
        context_docs = self.retriever.search(query, top_n=5)
        
        # Step 2: Gap Check (The Critic)
        # Use FAST mode for the critic (Logic check doesn't need DeepSeek Speciale)
        print("Running Gap Check...")
        context_str = self.format_context(context_docs)
        critic_prompt = f"""
        Does the retrieved context contain the specific answer to '{query}'? 
        If yes, output 'READY'. 
        If it mentions a specific file, table, or rule that is missing, output 'SEARCH: <new_query>'.
        Output ONLY 'READY' or 'SEARCH: ...'.
        """
        
        critic_response = self.llm.generate(
            system_prompt="You are a strict logic critic. Output only readiness status.", 
            user_prompt=f"Context:\n{context_str}\n\nUser Query: {query}\n\n{critic_prompt}",
            mode="FAST",
            temperature=0.0
        )
        
        # Step 3: Drill Down (Conditional)
        if "SEARCH:" in critic_response:
            new_query = critic_response.replace("SEARCH:", "").strip()
            print(f"Critic requested drill down: '{new_query}'")
            drill_docs = self.retriever.search(new_query, top_n=3)
            context_docs.extend(drill_docs)
            
            # Deduplicate
            seen_ids = set()
            unique_docs = []
            for d in context_docs:
                cid = d['chunk'].metadata.get('chunk_id')
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    unique_docs.append(d)
            context_docs = unique_docs
        else:
            print("Critic confirmed context is sufficient.")

        # Step 4: Synthesis (Reasoning)
        full_context = self.format_context(context_docs)
        system_prompt = "You are an expert AI Assistant. Synthesize a comprehensive strategy from the documentation. Cite sources."
        user_prompt = f"Context:\n{full_context}\n\nUser Query: \"{query}\""
        
        print("Synthesizing answer with Reasoning Model...")
        
        # Use DEEP mode (Gemini Pro -> DeepSeek Speciale)
        answer = self.llm.generate(
            system_prompt, 
            user_prompt,
            mode="DEEP",
            temperature=0.6
        )
            
        return answer, context_docs

    def query(self, user_query: str):
        """Main Entry Point."""
        route = self.route_query(user_query)
        print(f"Route selected: {route}")
        
        if route == "DEEP":
            return self.run_deep_track(user_query)
        else:
            return self.run_fast_track(user_query)

if __name__ == "__main__":
    runtime = RAGRuntime()
    
    q = "What is the policy for VIP escalation vs regular players?"
    print(f"\nUser Query: {q}")
    answer, docs = runtime.query(q)
    
    print("\n--- Final Answer ---")
    print(answer)