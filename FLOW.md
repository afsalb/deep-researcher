# Deep Researcher — User Flow & Architecture

This document details the step-by-step execution flow of the application, mapping user actions to agent methods and system logic.

## Phase A: Initial Research (Steps 1–8)

The goal is to produce a comprehensive report from a single topic.

**Step 1. User Input**
- User types a topic (e.g., "Future of Solid State Batteries") or clicks a quick-start chip.
- **System**: `app.py` captures `st.session_state.query_input`.

**Step 2. (Optional) Attachments**
- User uploads PDF/DOCX files.
- **System**: `utils.document_parser.parse_uploaded_file()` converts raw binary → text.

**Step 3. Start Search**
- User clicks "🚀 Start Search".
- **System**: Calls `agents.orchestrator.run_research(topic, uploaded_docs)`.

**Step 4. Query Decomposition**
- **Agent**: `Query Decomposer`
- **Method**: `decompose_query(topic) -> list[str]`
- **Logic**: Uses low-cost LLM to break the topic into 3-5 Google-optimized sub-queries.

**Step 5. Retrieval**
- **Agent**: `Retriever`
- **Method**: `retrieve_sources(sub_queries) -> list[dict]`
- **Logic**: 
  - Iterates through sub-queries.
  - Calls `TavilyClient.search()` for each.
  - Merges with `uploaded_docs`.
  - Deduplicates by URL.

**Step 6. Analysis**
- **Agent**: `Critical Analyzer`
- **Method**: `analyze_sources(docs) -> dict`
- **Logic**: 
  - Summarizes each document (max 200 words).
  - Scores credibility (0.0 - 1.0).
  - Identifies contradictions between sources.
  - **Output**: `summaries`, `contradictions`, `credible_docs`.

**Step 7. Insight Generation**
- **Agent**: `Insight Generator`
- **Method**: `generate_insights(analysis, topic) -> dict`
- **Logic**: Uses **high-reasoning model** (e.g., Claude 3.5 Sonnet) to find:
  - Key trends
  - Research gaps
  - Future hypotheses

**Step 8. Full Report Generation**
- **Agent**: `Report Builder`
- **Method**: `build_full_report(...) -> dict`
- **Logic**: 
  - Compiles all findings into a **detailed, long-form** Markdown report.
  - Generates BibTeX citations.
  - **Result**: Stored in `state.final_report`.

**Step 9. Executive Summary Generation**
- **Agent**: `Summarizer`
- **Method**: `generate_summary(full_report) -> str`
- **Logic**: 
  - Reads the full report from Step 8.
  - Generates a concise 2-3 paragraph summary.
  - **Result**: Stored in `state.summary`.

---

## Phase B: Chat Follow-up (Steps 10–15)

The goal is to answer specific questions about the report or new sub-topics interactively.

**Step 10. Chat Interface**
- Appears below the report result.
- Shows 3 auto-generated suggestion chips.

**Step 10. User Follow-up**
- User types a question or clicks a chip.
- Optionally uploads NEW files.

**Step 11. Intent Classification**
- **Agent**: `Chat Agent`
- **Method**: `classify_intent(msg, has_files) -> str`
- **Logic**: Uses **free model** (Llama 3) to classify into:
  - `answer_from_context`: Answer using existing report.
  - `needs_web_search`: Question requires external info.
  - `needs_attachment_analysis`: User uploaded new files.

**Step 12. Routing & Execution**
- **Route A (Context)**: 
  - Method: `answer_from_report(msg, report_context)`
  - Logic: Fast, cheap LLM call using report + source summaries.
- **Route B (Web Search)**:
  - Method: `retrieve_sources([msg])` → `analyze_sources(new_docs)` → `synthesize_answer()`
  - Logic: Performs a targeted single-query search.
- **Route C (Attachment)**:
  - Method: `parse_uploaded_file()` → `analyze_sources(new_docs)` → `synthesize_answer()`
  - Logic: Reads new file and incorporates it into the answer.

**Step 13. Suggestions**
- **Agent**: `Follow-up Agent`
- **Method**: `suggest_followups(msg, answer) -> list[str]`
- **Logic**: Generates 3 new relevant questions based on the latest turn.

**Step 14. Display**
- Chat bubble appears with the answer.
- "New Search" or "File Analysis" badges shown if Routes B or C were used.
- New suggestion chips update.
