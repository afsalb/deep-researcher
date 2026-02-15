# Deep Researcher — Architecture & Execution Flow

This document provides a comprehensive technical analysis of the **Deep Researcher** application. It details the system architecture, state management, and the step-by-step execution flows for both the research and chat pipelines.

## 1. System Architecture

The application is built using a **Micro-Agent Architecture** orchestrated by **LangGraph**. The frontend is a **Streamlit** application that interacts with these agents via a shared state.

### Core Components
- **Frontend (`app.py`)**: Handles user I/O, session state, and renders the UI (Glassmorphic Material 3 design).
- **Orchestrator (`agents/orchestrator.py`)**: Manages the LangGraph workflows (`StateGraph`) and coordinates data flow between agents.
- **Agents (`agents/`)**: Specialized modules responsible for specific cognitive tasks (Decomposition, Retrieval, Analysis, etc.).
- **Tools (`utils/`)**: Helper functions for LLM calls, web search (Tavily), and document parsing.

---

## 2. Shared State Management

The application uses **LangGraph State** (TypedDict) to maintain context across agent steps. This ensures that all agents have access to the accumulating knowledge.

### Research Graph State (`GraphState`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `topic` | `str` | The original user query. |
| `sub_queries` | `list[str]` | Decomposed search queries. |
| `retrieved_docs` | `list[dict]` | Raw content from web + uploads. |
| `analysis_notes` | `dict` | Summaries and credibility scores. |
| `insights` | `dict` | Hypotheses, trends, and gaps. |
| `final_report` | `str` | The generated Markdown report. |
| `summary` | `str` | The executive summary. |

---

## 3. Deep Research Pipeline (Steps 1–6)

This pipeline transforms a simple user topic into a comprehensive research report.

### **Step 1: Query Decomposition**
- **Agent**: `Query Decomposer` (`agents/query_decomposer.py`)
- **Input**: User Topic (e.g., "Future of Solid State Batteries")
- **Action**: Uses a low-cost LLM to break the topic into 3-5 specific, google-optimized sub-questions.
- **Goal**: Ensure broad coverage of the topic.
- **Output**: `state['sub_queries']` = `["solid state battery distinct advantages", "manufacturing challenges of SSBs", ...]`

### **Step 2: Retrieval (Web & Local)**
- **Agent**: `Retriever` (`agents/retriever.py`)
- **Input**: `sub_queries` + `uploaded_docs`
- **Action**: 
    1. **Web Search**: Queries **Tavily API** for each sub-query (max 5 results per query).
    2. **Local**: Merges user-uploaded documents (PDF/DOCX).
    3. **Deduplication**: Removes duplicate URLs to prevent context waste.
- **Output**: `state['retrieved_docs']` (List of raw text content and metadata).

### **Step 3: Critical Analysis**
- **Agent**: `Critical Analyzer` (`agents/analyzer.py`)
- **Input**: `retrieved_docs`
- **Action**:
    - Summarizes each document (max 200 words).
    - **Scores Credibility** (0.0 - 1.0) based on source reputation and content quality.
    - Identifies **Contradictions** between sources.
- **Goal**: Filter out noise and identify conflicting information.
- **Output**: `state['analysis_notes']` (`summaries`, `contradictions`, `credible_docs`).

### **Step 4: Insight Generation**
- **Agent**: `Insight Generator` (`agents/insight_generator.py`)
- **Input**: `analysis_notes`
- **Action**: Uses a **High-Reasoning Model** (e.g., Claude 3.5 Sonnet / GPT-4o) to synthesize:
    - **Key Trends**: Where is the field heading?
    - **Hypotheses**: Testable predictions.
    - **Research Gaps**: What is missing from current literature?
- **Output**: `state['insights']`.

### **Step 5: Full Report Generation**
- **Agent**: `Report Builder` (`agents/report_builder.py`)
- **Input**: `topic` + `analysis_notes` + `insights` + `retrieved_docs`
- **Action**:
    - Compiles a long-form Markdown report (Structure: Executive Summary, Findings, Analysis, Insights, Conclusion).
    - Generates **BibTeX** citations for all used sources.
- **Output**: `state['final_report']` and `state['bibtex']`.

### **Step 6: Summarization**
- **Agent**: `Summarizer` (`agents/report_builder.py`)
- **Action**: Condenses the potentially massive full report into a 2-3 paragraph Executive Summary for quick reading.
- **Output**: `state['summary']`.

---

## 4. Chat Pipeline (Follow-up)

After the report is generated, the user can ask follow-up questions. This pipeline uses a separate **Chat Graph**.

### **Step 1: Intent Classification**
- **Agent**: `Chat Classifier` (`agents/chat_agent.py`)
- **Input**: User Message + Chat History + (Optional) New Attachments
- **Action**: Determines the routing logic:
    - `answer_from_context`: Answer using the existing report (Fast/Cheap).
    - `needs_web_search`: The question requires new external information.
    - `needs_attachment_analysis`: The user uploaded a new file to analyze.

### **Step 2: Routing & Execution**

#### **Route A: Answer from Context**
- **Action**: Passes the `final_report` and `source_summaries` to the LLM.
- **Prompt**: "Answer solely based on the provided research context."

#### **Route B: Web Search (Knowledge Expansion)**
- **Action**: 
    1. Transforms the user message into a search query.
    2. Calls `retrieve_sources()` (same tool as Step 2 above).
    3. Calls `analyze_sources()` on the *new* findings.
    4. **Synthesizes** an answer combining the new findings with the user's question.

#### **Route C: Attachment Analysis**
- **Action**:
    1. Parses the new uploaded file (`utils/document_parser.py`).
    2. Runs `analyze_sources()` on the file content.
    3. Synthesizes an answer based on the document.

### **Step 3: Follow-up Suggestions**
- **Agent**: `Follow-up Agent` (`agents/followup_agent.py`)
- **Action**: Generates 3 relevant questions based on the last answer to encourage deeper exploration.
- **Output**: Chips displayed in the UI (e.g., "Tell me more about X").

---

## 5. File Structure Map

```text
root/
├── app.py                  # Main Streamlit Entry Point
├── FLOW.md                 # This Architecture Document
├── agents/                 # Logic Core
│   ├── orchestrator.py     # LangGraph Setup (Research & Chat graphs)
│   ├── query_decomposer.py # Step 1
│   ├── retriever.py        # Step 2
│   ├── analyzer.py         # Step 3
│   ├── insight_generator.py# Step 4
│   ├── report_builder.py   # Step 5 & 6
│   ├── chat_agent.py       # Chat Logic & Routing
│   └── followup_agent.py   # Suggestion Generation
├── utils/                  # Shared Tools
│   ├── llm_client.py       # Unified LLM API Wrapper
│   ├── tavily_client.py    # Search API Wrapper
│   ├── document_parser.py  # PDF/DOCX -> Text
│   └── cost_tracker.py     # Token Usage Monitoring
└── config/                 # Configuration
    └── models.py           # Model definitions & parameters
```
