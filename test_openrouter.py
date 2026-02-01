import time
import os
from rag_runtime import LLMWrapper

# --- Configuration ---
CONTEXT_SIZE = 4500
MODEL_MODE = "FAST"  # "FAST" = deepseek-v3.2, "DEEP" = deepseek-v3.2-speciale

def run_test():
    print(f"--- Starting DeepSeek V3.2 Context Test ({CONTEXT_SIZE} chars) ---")
    
    # 1. Initialize Wrapper
    llm = LLMWrapper()
    
    # 2. FORCE FAILOVER: Disable Gemini client to ensure we hit OpenRouter (DeepSeek)
    print("⚠️  Forcing Failover to OpenRouter (DeepSeek)...")
    llm.gemini_client = None 

    if not llm.openrouter_client:
        print("❌ Error: OPENROUTER_API_KEY not found or client failed to init.")
        return

    # 3. Generate Dummy Context (4500 chars)
    # Repeating a paragraph to hit the target size
    base_text = (
        "DeepSeek-V3 is a powerful Mixture-of-Experts (MoE) language model with 671B parameters. "
        "It uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures. "
        "This text is used to test the context window handling capabilities of the API. "
    )
    long_context = (base_text * 50)[:CONTEXT_SIZE]
    print(f"✅ Generated Context: {len(long_context)} characters")

    # 4. Define Prompts
    system_prompt = "You are a helpful AI. Analyze the context and answer the user query."
    user_prompt = f"Context:\n{long_context}\n\nUser Query: Summarize the key technical architecture mentioned in the text."

    # 5. Run Generation
    print(f"🚀 Sending request to DeepSeek (Mode: {MODEL_MODE})...")
    start_time = time.time()
    
    try:
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=MODEL_MODE,
            temperature=0.0
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*40)
        print("✅ RESPONSE RECEIVED")
        print(f"⏱️  Time Taken: {elapsed:.2f}s")
        print("="*40)
        print(response)
        print("="*40)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == "__main__":
    run_test()