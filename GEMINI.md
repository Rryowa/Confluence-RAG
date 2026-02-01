# Gemini RAG Development Plan - Final Master Protocol

This is the **Final Master Protocol** for the Confluence RAG system. It integrates every constraint, optimization, and architectural decision into a single executable plan.

## Phase 0:
 Storage "Toy" Implementation**

* **The Component:** `LocalFileStore`.
* **The Issue:** This stores documents as loose files on your disk. It is not ACID-compliant. If your ingestion script crashes halfway through updating a Confluence page, your `ChromaDB` (vectors) will point to a `LocalFileStore` ID that either doesn't exist or contains old text. This "state drift" causes hallucinations where the bot quotes text that no longer exists.
* **The Fix:** Use **SQLite** (via `SQLStore` in LangChain) or just put the text inside Chroma's metadata (if docs are <40kb). Do not use file-system storage for a "State-of-the-Art" system.


## Phase 1: Advanced Data Engineering (Sanitization & Chunking)

*Goal: Feed the model high-signal text, not HTML noise.*

**Implementation Update (2026-02-01):**
*   Switched from `ConfluenceLoader` to **Direct Atlassian API** (`body.storage`) + `markdownify`.
*   *Reason:* `ConfluenceLoader` stripped headers and context. Direct API allows custom `markdownify` settings (`heading_style="ATX"`, `wrap=False`).

1.  **Sanitization (Regex Rules)**
    *   **Nuke Images:** Remove `!\[.*?\]\(.*?\)` (Images = noise).
    *   **Hex Colors:** Remove standalone hex codes (e.g., `#DEEBFF`) often left by background macros.
    *   **Clean Links:**
        *   Remove angle brackets: `<https://...>` -> `https://...`.
        *   Strip Tracking: Remove `?atlOrigin=...` params.
        *   Simplify: `[Text](url)` -> `Text` (if standard link).
    *   **Unwrap Expanders:** Remove `<details>`/`<summary>` tags but **keep the text inside**.
    *   **Flatten Tables:** Convert `| Header | Value |` -> `Header: Value`.
    *   **Noise Reduction:**
        *   **Visual Anchors:** Remove "(pic above)", "This is what you are going to see:".
        *   **Procedural:** Remove "File -> Make a Copy".
        *   **Quiz Feedback:** Remove "Answer 86 is correct!", "If you got this, congrats!".
    *   **High-Value Extraction:**
        *   **Bold Text:** Extract `**text**` and inject as "Keywords" in metadata (Strategy A).

2.  **"Sticky" Macro Logic (Crucial for Context)**
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
   * **Embedding:** `Qwen3-embedding-0.6b` (~1.2GB VRAM).
   * **Reranker:** `jina-reranker-v3-turbo-en` (~1.2GB VRAM).
   * **Keyword Search:** `RankBM25` (Lightweight sparse index).
   * **Loader:** `transformers` with `attn_implementation="flash_attention_2"`.
   * **Hard Constraint:** No `llama.cpp` / GGUF / Int8.

## Phase 3: The "Deep Reasoning" Runtime (The Brain)

*Goal: Handle complex strategy questions without crashing. Minimize cost by using smaller models for plumbing.*

6. **The Router (Step 0)**
   * **Model:** `gpt-4o-mini` or similar.
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
    You are an expert query dispatcher for a corporate Confluence knowledge base.
Your only job is to classify the User Query into one of two routes: "FAST" or "DEEP".

LOGIC:
- ROUTE: FAST
  - Trigger: The user asks for a simple fact, a single policy, a "how-to" guide, or a definition.
  - Examples: "How do I reset my password?", "What is the holiday allowance?", "Show me the VPN guide."
  
- ROUTE: DEEP
  - Trigger: The user asks for a strategy, a comparison between two rules, a "best way" to do something involving multiple constraints, or complex reasoning (e.g., "Upsell strategy based on limitations").
  - Examples: "How do I handle a VIP client who is angry about the bonus limit?", "Compare the sick leave policy for contractors vs employees."

OUTPUT FORMAT:
Return ONLY a valid JSON object. Do not add explanations.
{
    "route": "FAST" | "DEEP",
    "reasoning": "Short 1-sentence explanation"
}

User Query: "{user_query}"

12. **The Gap Check Prompt**
    * *"Do I have enough information to answer the user fully? The user asked '{user_query}'. I found context about '{context_summary}', but do I know '{specific_detail}'? Output 'READY' or 'SEARCH: <target>'."*

13. **The RAG Generation Prompt**
    You are a Confluence Knowledge Assistant. You are an expert at synthesizing technical documentation into clear, actionable answers.

CONTEXT:
You have been provided with retrieved excerpts from our internal documentation. Each excerpt has a [Source ID].
{retrieved_chunks}

INSTRUCTIONS:
1. **Answer the User Query** primarily using the provided context.
2. **Handle Conflicts:** If two chunks provide conflicting information (e.g., "Chunk 1 says Limit is $50" vs "Chunk 2 says Limit is $100"), explicitly mention the conflict and look for conditions (e.g., "VIP vs Regular") to resolve it. Do not guess.
3. **Cite Sources:** You MUST cite the [Source ID] for every fact you state. (e.g., "The limit is $50 [Source: 12]").
4. **Unknowns:** If the context does not contain the answer, state: "I cannot find this specific information in the provided documentation." Do not make up internal policies.
5. **Formatting:** Use clear headers and bullet points. If the user asked for a strategy, structure the answer as a step-by-step guide.

User Query: "{user_query}"

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
