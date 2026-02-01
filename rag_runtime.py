import os
import argparse
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

GEMINI_FAST_MODEL = "gemini-3-flash-preview"
GEMINI_DEEP_MODEL = "gemini-3-pro-preview"
OR_FAST_MODEL = "deepseek/deepseek-v3.2"
OR_DEEP_MODEL = "deepseek/deepseek-v3.2"

class LLMWrapper:
    def __init__(self):
        self.gemini_client = None
        self.openrouter_client = None
        
        if GEMINI_API_KEY and genai:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"Primary Client: Gemini ({GEMINI_FAST_MODEL} / {GEMINI_DEEP_MODEL})")
            except Exception as e:
                print(f"Failed to init Gemini: {e}")

        if OPENROUTER_API_KEY:
            try:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )
                print(f"Failover Client: OpenRouter ({OR_FAST_MODEL} / {OR_DEEP_MODEL})")
            except Exception as e:
                print(f"Failed to init OpenRouter: {e}")

    def _call_gemini(self, model: str, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> str:
        if not self.gemini_client: raise ImportError("Gemini client missing")
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
            system_instruction=system_prompt
        )
        response = self.gemini_client.models.generate_content(
            model=model, contents=[user_prompt], config=config, timeout=120 
        )
        return response.text

    def _call_openrouter(self, model: str, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool) -> str:
        if not self.openrouter_client: raise ImportError("OpenRouter client missing")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        params = {"model": model, "messages": messages, "temperature": temperature, "extra_body":{"reasoning": {"enabled": True}}}
        if json_mode: params["response_format"] = {"type": "json_object"}
        
        print(f"   (Waiting for OpenRouter response from {model} [Timeout: 300s]...)")
        return self.openrouter_client.chat.completions.create(**params, timeout=300.0).choices[0].message.content

    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False, mode: str = "FAST", temperature: float = 0.0) -> str:
        if mode == "DEEP":
            primary, failover = GEMINI_DEEP_MODEL, OR_DEEP_MODEL
        else:
            primary, failover = GEMINI_FAST_MODEL, OR_FAST_MODEL

        # --- DEBUG PRINTS ---
        print("\n" + "="*40)
        print(f"🔍 DEBUG: SENDING TO {mode} MODEL")
        print("="*40)
        print(f"--- SYSTEM PROMPT ---\n{system_prompt}\n")
        print(f"--- USER PROMPT (Truncated first 500 chars) ---\n{user_prompt[:1000]}...\n[... {len(user_prompt)} chars total ...]")
        print("="*40 + "\n")

        if self.gemini_client:
            try:
                return self._call_gemini(primary, system_prompt, user_prompt, temperature, json_mode)
            except Exception as e:
                print(f"[Gemini Error]: {e}. Switching to OpenRouter...")
        
        if self.openrouter_client:
            try:
                return self._call_openrouter(failover, system_prompt, user_prompt, temperature, json_mode)
            except Exception as e:
                print(f"[OpenRouter Error]: {e}")
                return "ERROR"
        return "MOCKED_RESPONSE"

class RAGRuntime:
    def __init__(self):
        print("Initializing RAG Runtime...")
        self.retriever = HybridRetriever()
        self.llm = LLMWrapper()

    def route_query(self, query: str) -> str:
        # Route logic (simplified for brevity)
        return "FAST" 

    def format_context(self, retrieved_results: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, res in enumerate(retrieved_results):
            chunk = res['chunk']
            parent = res['parent']
            context_str += f"\n--- [Source ID: {i+1}] ---\nTitle: {parent.metadata.get('title') if parent else 'Unknown'}\nContent:\n{chunk.page_content}"
        return context_str

    def run_fast_track(self, query: str):
        print("--- Executing FAST Track ---")
        results = self.retriever.search(query, top_n=3)
        context = self.format_context(results)
        
        system_prompt = '''You are a Confluence Knowledge Assistant. You are an expert at synthesizing technical documentation into clear, actionable answers.

CONTEXT:
You have been provided with retrieved excerpts from our internal documentation. Each excerpt has a [Source ID].
{retrieved_chunks}

INSTRUCTIONS:
1. **Answer the User Query** primarily using the provided context.
2. **Handle Conflicts:** If two chunks provide conflicting information (e.g., "Chunk 1 says Limit is $50" vs "Chunk 2 says Limit is $100"), explicitly mention the conflict and look for conditions (e.g., "VIP vs Regular") to resolve it. Do not guess.
3. **Cite Sources:** You MUST cite the [Source ID] for every fact you state. (e.g., "The limit is $50 [Source: 12]").
4. **Unknowns:** If the context does not contain the answer, state: "I cannot find this specific information in the provided documentation." Do not make up internal policies.
5. **Formatting:** Use clear headers and bullet points. If the user asked for a strategy, structure the answer as a step-by-step guide.

User Query: "{user_query}'''
        user_prompt = f"Context:\n{context}\n\nUser Query: \"{query}\""
        
        answer = self.llm.generate(system_prompt, user_prompt, mode="FAST", temperature=0.0)
        return answer, results

    def run_deep_track(self, query: str):
        print("--- Executing DEEP Track ---")
        context_docs = self.retriever.search(query, top_n=5)
        
        # Gap Check (Critic)
        context_str = self.format_context(context_docs)
        critic_prompt = f"Does the context answer '{query}' and prioritized the cause over the consequence? ( Output 'READY' or 'SEARCH: <query>'."
        
        # We print debug info for the critic too
        print("--- DEBUG: RUNNING CRITIC ---")
        critic_response = self.llm.generate(
            "You are a strict logic critic.", 
            f"Context:\n{context_str}\n\nUser Query: {query}\n\n{critic_prompt}",
            mode="FAST", temperature=0.0
        )
        
        if "SEARCH:" in critic_response:
            new_query = critic_response.replace("SEARCH:", "").strip()
            print(f"Critic requested drill down: '{new_query}'")
            drill_docs = self.retriever.search(new_query, top_n=3)
            context_docs.extend(drill_docs)
        
        # Synthesis
        full_context = self.format_context(context_docs)
        system_prompt = '''You are a Confluence Knowledge Assistant. You are an expert at synthesizing technical documentation into clear, actionable answers.

CONTEXT:
You have been provided with retrieved excerpts from our internal documentation. Each excerpt has a [Source ID].
{retrieved_chunks}

INSTRUCTIONS:
1. **Answer the User Query** primarily using the provided context.
2. **Handle Conflicts:** If two chunks provide conflicting information (e.g., "Chunk 1 says Limit is $50" vs "Chunk 2 says Limit is $100"), explicitly mention the conflict and look for conditions (e.g., "VIP vs Regular") to resolve it. Do not guess.
3. **Cite Sources:** You MUST cite the [Source ID] for every fact you state. (e.g., "The limit is $50 [Source: 12]").
4. **Unknowns:** If the context does not contain the answer, state: "I cannot find this specific information in the provided documentation." Do not make up internal policies.
5. **Formatting:** Use clear headers and bullet points. If the user asked for a strategy, structure the answer as a step-by-step guide.

User Query: "{user_query}'''
        user_prompt = f"Context:\n{full_context}\n\nUser Query: \"{query}\""
        
        print("Synthesizing answer with Reasoning Model...")
        answer = self.llm.generate(system_prompt, user_prompt, mode="DEEP", temperature=0.6)
        return answer, context_docs

    def query(self, user_query: str, force_mode: Optional[str] = None):
        if force_mode:
            route = force_mode
            print(f"Route forced: {force_mode}")
        else:
            route = "FAST" # Default fallback
            
        if route == "DEEP":
            return self.run_deep_track(user_query)
        else:
            return self.run_fast_track(user_query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=str, required=True, choices=["true", "false"], help="Enable deep reasoning.")
    args = parser.parse_args()
    
    runtime = RAGRuntime()
    q = '''Create a step-by-step guide for handling a 'High Priority' regular player who is threatening a refund due to a technical error. Include required tags and escalation paths
'''
    print(f"\nUser Query: {q}")
    
    mode = "DEEP" if args.reasoning == "true" else "FAST"
    answer, docs = runtime.query(q, force_mode=mode)
    
    print("\n--- Final Answer ---")
    print(answer)