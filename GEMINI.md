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

4. **Chunking**
   * **Method:** Recursive Header Split (H2, H3).
   * **Size:** 512–1024 tokens.
   * **Metadata:** Append `Page Title > Section Header` to text.

## Phase 2: The 0.6B Inference Stack (Hardware Optimized)

*Goal: Speed and Low VRAM.*

5. **Models (FP16 & Native PyTorch)**
   * **Embedding:** `Qwen2.5-0.5B-Instruct` (~1.2GB VRAM).
   * **Reranker:** `jina-reranker-v3-turbo-en` (~1.2GB VRAM).
   * **Loader:** `transformers` with `attn_implementation="flash_attention_2"`.
   * **Hard Constraint:** No `llama.cpp` / GGUF / Int8.

## Phase 3: The "Deep Reasoning" Runtime (The Brain)

*Goal: Handle complex strategy questions without crashing.*

6. **The Router (Step 0)**
   * **Logic:** Classify user intent.
   * **Prompt:** *"Is this a simple lookup (Fast) or a multi-factor strategy/reasoning task (Deep)?"*

7. **Track A: Fast Lookup (Simple)**
   * **Retrieval:** Top **50** (High Recall).
   * **Rerank:** Top 3.

8. **Track B: Deep Decomposition (Complex)**
   * **Decomposition:** Break query into 2–4 sub-questions (e.g., "Rules," "Limits," "Exceptions").
   * **Step-Back:** Generate 1 abstract context query.
   * **Execution:** Run *Fast Lookup* for each sub-question.
   * **The "Low-K" Optimization:** Retrieve only **Top 20** per sub-question.
   * *Why:* 3 sub-questions × 20 docs = 60 reranks (Manageable). 3 × 50 = 150 reranks (Too slow).
   * **Synthesis:** Gemini aggregates facts into a strategy.

## Phase 4: Prompt Engineering Strategy

*Goal: Ensure the LLM uses the tools correctly.*

9. **The Router Prompt**
   * *"You are an expert dispatcher. Your only job is to output 'FAST' or 'DEEP'. 'DEEP' is for queries asking for comparisons, strategies, or combined constraints. 'FAST' is for single-fact lookups."*

10. **The Decomposition Prompt**
    * *"You are a strategy analyst. Break this complex request into a JSON list of atomic search queries. Separate the 'Rules' from the 'Exceptions'. Do not answer yet, just list the queries."*

11. **The RAG Generation Prompt**
    * *"Using the following retrieved context (Source ID: X), answer the user. If the context contains conflicting rules (e.g., VIP vs Regular), explicitly state the distinction."*

## Phase 5: Operational Stability (The "Squeeze")

12. **Child-Only Reranking**
    * **Rule:** Pass only the **500-token chunk** to the Reranker.
    * **Action:** Only fetch the full "Parent" document *after* the chunk wins the ranking.

13. **Dynamic Batching (OOM Protection)**
    * **Logic:** `try/except` block on `OutOfMemoryError`.
    * **Start:** Batch 128 (Embed), Batch 32 (Rerank Chunks).
    * **Fail:** Empty Cache -> Batch / 2 -> Retry.

## Phase 6: Quality Control (The Final Check)

14. **The Golden Dataset**
    * **Dataset:** 50 Real Q&A pairs with correct Chunk IDs.
    * **Metric:** **Hit Rate @ 5** (Target >90%).
    * **Rule:** If you change the prompt or chunk size, run this test. If Hit Rate drops, revert.
