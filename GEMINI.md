# Gemini RAG Development Plan - Final Master Protocol

This is the **Final Master Protocol** for the Confluence RAG system. It integrates every constraint, optimization, and architectural decision into a single executable plan.

## Phase 1: Advanced Data Engineering (Sanitization & Chunking)

*Goal: Feed the model high-signal text, not HTML noise.*

1. **Sanitization (Regex Rules)**
   * **Nuke Images:** Remove `!\[.*?\]\(.*?\)` (Images = noise).
   * **Unwrap Expanders:** Remove `<details>`/`<summary>` tags but **keep the text inside**.
   * **Flatten Tables:** Convert `| Header | Value |` -> `Header: Value`.
   * **Simplify Links:** `[Text](url)` -> `Text`.

2. **"Sticky" Macro Logic (Crucial for Context)**
   * **Rule:** Detect `confluence-information-macro` (Info/Note boxes).
   * **Action:** **Prepend** the content of the macro to the *immediately following* paragraph before chunking.
   * *Why:* Prevents a warning like "Important: Only for VIPs" from being orphaned in a separate chunk from the instructions it modifies.

3. **Template Isolation Strategy**
   * **Rule:** Detect email templates (blocks starting with "Dear...", signatures).
   * **Action:** Force a **hard split** before and after the template. Treat the template as a **distinct atomic chunk**.
   * *Why:* Templates are high-value answers. You want the Reranker to see the pure template, not mixed with other rules.

4. **Chunking & Metadata**
   * **Method:** Recursive Header Split (H2, H3).
   * **Size:** 512–1024 tokens.
   * **Metadata:** 
     * Append `Page Title > Section Header` to text.
     * **Crucial:** Capture `last_updated` date (for time decay logic).

## Phase 2: The 0.6B Inference Stack (Hardware Optimized)

*Goal: Speed and Low VRAM.*

5. **Models (FP16 & Native PyTorch)**
   * **Embedding:** `Qwen2.5-0.5B-Instruct` (~1.2GB VRAM).
   * **Reranker:** `jina-reranker-v3-turbo-en` (~1.2GB VRAM).
   * **Keyword Search:** `RankBM25` (Lightweight sparse index).
   * **Loader:** `transformers` with `attn_implementation="flash_attention_2"`.
   * **Hard Constraint:** No `llama.cpp` / GGUF / Int8.

## Phase 3: The "Deep Reasoning" Runtime (The Brain)

*Goal: Handle complex strategy questions without crashing. Minimize cost by using smaller models for plumbing.*

6. **The Router (Step 0)**
   * **Model:** `gpt-4o-mini`.
   * **Logic:** Classify user intent.
   * **Prompt:** *"Is this a simple lookup (Fast) or a multi-factor strategy/reasoning task (Deep)?"*

7. **Hybrid Retrieval Core (Used by both tracks)**
   * **Vector Search:** Fetch Top 25.
   * **Keyword Search (BM25):** Fetch Top 25.
   * **Deduplicate:** Combine lists (remove duplicates).
   * **Rerank:** Pass the combined list (e.g., ~40 docs) to `jina-reranker-v3`.
   * *Why:* BM25 ensures exact keyword matches (like error codes) survive to the Reranker even if vector semantics miss them.

8. **Track A: Fast Lookup (Simple)**
   * **Action:** Run Hybrid Retrieval.
   * **Filter:** Return Top 3 Reranked results.
   * **Answer:** Immediate generation via `gpt-4o-mini`.

9. **Track B: The "Check & Drill" Protocol (Deep)**
   * **Step 1: The "Broad Sweep" (Parallel Decomposition)**
     * **Model:** `gpt-4o-mini`.
     * Break query into sub-questions (e.g., "Error 509 meaning", "Payment API limits").
     * Run **Hybrid Retrieval** for each.
   * **Step 2: The "Gap Check" (The Critic)**
     * **Model:** `gpt-4o-mini`.
     * **Prompt:** *"Does the retrieved context contain the specific answer to '{query}'? If yes, output 'READY'. If it mentions a specific file, table, or rule that is missing, output 'SEARCH: <new_query>'."*
   * **Step 3: The "Drill Down" (Conditional)**
     * If `SEARCH: ...`: Execute **one** single targeted lookup for that specific term.
     * Append result to context.
   * **Step 4: Synthesis (Reasoning)**
     * **Model:** `DeepSeek-R1`.
     * **Logic:** Pass the full consolidated context to the reasoning model. You only pay the "Reasoning Tax" here.

10. **Scoring Logic: Time-Weighted Decay**
    * **Application:** Post-Reranking or during Scoring.
    * **Formula:** `Final_Score = Reranker_Score + (Days_Since_Update * -0.0001)`
    * **Hard Filter (Deep Track Only):** Discard docs older than 2 years unless metadata tag contains "Evergreen".

## Phase 4: Prompt Engineering Strategy

*Goal: Ensure the LLM uses the tools correctly.*

11. **The Router Prompt**
    * *"You are an expert dispatcher. Your only job is to output 'FAST' or 'DEEP'. 'DEEP' is for queries asking for comparisons, strategies, or combined constraints. 'FAST' is for single-fact lookups."*

12. **The Gap Check Prompt**
    * *"Do I have enough information to answer the user fully? The user asked '{user_query}'. I found context about '{context_summary}', but do I know '{specific_detail}'? Output 'READY' or 'SEARCH: <target>'."*

13. **The RAG Generation Prompt**
    * *"Using the following retrieved context (Source ID: X), answer the user. If the context contains conflicting rules (e.g., VIP vs Regular), explicitly state the distinction."*

## Phase 5: Operational Stability (The "Squeeze")

14. **Child-Only Reranking**
    * **Rule:** Pass only the **500-token chunk** to the Reranker.
    * **Action:** Only fetch the full "Parent" document *after* the chunk wins the ranking.

15. **Dynamic Batching (OOM Protection)**
    * **Logic:** `try/except` block on `OutOfMemoryError`.
    * **Start:** Batch 128 (Embed), Batch 32 (Rerank Chunks).
    * **Fail:** Empty Cache -> Batch / 2 -> Retry.

## Phase 6: Quality Control (The Final Check)

16. **The Golden Dataset**
    * **Dataset:** 50 Real Q&A pairs with correct Chunk IDs.
    * **Metric:** **Hit Rate @ 5** (Target >90%).
    * **Rule:** If you change the prompt or chunk size, run this test. If Hit Rate drops, revert.
