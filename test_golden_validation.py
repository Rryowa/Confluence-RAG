import rag_runtime
from rag_runtime import RAGRuntime
import time

# --- Configuration ---
# Override model to a more stable/higher-quota model for testing
# (gemini-3-pro-preview hits free-tier quota too quickly for interactive testing)
rag_runtime.DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Golden Dataset (Mini)
TEST_CASES = [
    {
        "id": "FAST_01",
        "query": "What is the specific timeframe for escalating a pending deposit for a VIP player?",
        "expected_facts": ["2 business days", "VIP"],
        "track_hint": "FAST" 
    },
    {
        "id": "DEEP_01",
        "query": "Create a step-by-step guide for handling a 'High Priority' regular player who is threatening a refund due to a technical error. Include required tags and escalation paths.",
        "expected_facts": ["PSP_High Priority", "out of turn", "technical error", "refund threat"],
        "track_hint": "DEEP"
    }
]

def run_validation():
    print(f"--- Starting Golden Dataset Validation (Model: {rag_runtime.DEFAULT_GEMINI_MODEL}) ---\n")
    runtime = RAGRuntime()
    
    for case in TEST_CASES:
        print(f"\n>>> Test Case: {case['id']} ({case['track_hint']})")
        print(f"Query: {case['query']}")
        
        start_time = time.time()
        # We let the router decide, but we expect it to match hint. 
        # Or we can force it? Let's test the Router too.
        answer, docs = runtime.query(case['query'])
        duration = time.time() - start_time
        
        print(f"\n[Generated Answer ({duration:.2f}s)]:")
        print(answer)
        print("-" * 40)
        
        # Simple Keyword Validation
        print("[Validation]:")
        missing = []
        for fact in case['expected_facts']:
            if fact.lower() not in answer.lower():
                missing.append(fact)
        
        if not missing:
            print("✅ PASS: All key facts present.")
        else:
            print(f"❌ FAIL: Missing facts: {missing}")
            
    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    run_validation()
