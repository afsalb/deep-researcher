"""
LangGraph Orchestrator — coordinates the 5-agent research pipeline.

Pipeline: Query Decomposer → Retriever → Analyzer → Insight Generator → Report Builder

Uses LangGraph StateGraph with a TypedDict for proper state management.
Each node returns partial updates that get merged correctly.
"""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.query_decomposer import decompose_query
from agents.retriever import retrieve_sources
from agents.analyzer import analyze_sources
from agents.insight_generator import generate_insights
from agents.report_builder import build_full_report, generate_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph State (TypedDict with reducers for proper merging)
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """LangGraph state — uses Annotated reducers so partial node returns
    get merged properly instead of overwriting the whole state."""
    topic: str
    sub_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    uploaded_docs: list[dict[str, Any]]
    analysis_notes: dict[str, Any]
    insights: dict[str, Any]
    final_report: str
    summary: str
    bibtex: str
    errors: list[str]
    agent_logs: Annotated[list[dict[str, str]], operator.add]  # additive merge


# ---------------------------------------------------------------------------
# Output dataclass — what app.py consumes
# ---------------------------------------------------------------------------

@dataclass
class ResearchState:
    """Final output of the research pipeline."""
    topic: str = ""
    sub_queries: list[str] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    analysis_notes: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    final_report: str = ""
    bibtex: str = ""
    errors: list[str] = field(default_factory=list)
    agent_logs: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level progress callback
# ---------------------------------------------------------------------------

_progress_callback: Optional[Callable[[str, float], None]] = None


# ---------------------------------------------------------------------------
# Pipeline nodes — each returns ONLY the keys it changes
# ---------------------------------------------------------------------------

def node_decompose(state: GraphState) -> dict:
    """Node 1: Decompose topic into sub-queries."""
    topic = state.get("topic", "")

    if _progress_callback:
        _progress_callback("🔍 Decomposing query…", 0.10)

    try:
        sub_queries = decompose_query(topic)
    except Exception as e:
        logger.error("Decomposition error: %s", e)
        sub_queries = [topic]

    return {
        "sub_queries": sub_queries,
        "agent_logs": [{"agent": "query_decomposer", "message": f"Decomposed into {len(sub_queries)} sub-queries"}],
    }


def node_retrieve(state: GraphState) -> dict:
    """Node 2: Retrieve web sources."""
    sub_queries = state.get("sub_queries", [])
    uploaded_docs = state.get("uploaded_docs", [])

    if _progress_callback:
        _progress_callback("📚 Retrieving sources…", 0.30)

    try:
        docs = retrieve_sources(sub_queries, uploaded_docs=uploaded_docs)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = list(uploaded_docs) if uploaded_docs else []

    return {
        "retrieved_docs": docs,
        "agent_logs": [{"agent": "retriever", "message": f"Retrieved {len(docs)} sources"}],
    }


def node_analyze(state: GraphState) -> dict:
    """Node 3: Analyze and validate sources."""
    docs = state.get("retrieved_docs", [])

    if _progress_callback:
        _progress_callback("🧪 Analyzing sources…", 0.50)

    try:
        analysis = analyze_sources(docs)
    except Exception as e:
        logger.error("Analysis error: %s", e)
        analysis = {"summaries": [], "contradictions": [], "credible_docs": list(docs)}

    return {
        "analysis_notes": analysis,
        "agent_logs": [{"agent": "analyzer", "message": f"Analyzed {len(docs)} documents"}],
    }


def node_insights(state: GraphState) -> dict:
    """Node 4: Generate insights."""
    analysis = state.get("analysis_notes", {})
    topic = state.get("topic", "")

    if _progress_callback:
        _progress_callback("💡 Generating insights…", 0.70)

    try:
        insights = generate_insights(analysis, topic)
    except Exception as e:
        logger.error("Insight error: %s", e)
        insights = {"insights": [], "hypotheses": [], "trends": [], "gaps": []}

    return {
        "insights": insights,
        "agent_logs": [{"agent": "insight_generator", "message": "Generated insights"}],
    }


def node_report_builder(state: GraphState) -> dict:
    """Generate the detailed full report."""
    if _progress_callback:
        _progress_callback("📝 Writing full report...", 0.85)

    topic = state.get("topic", "")
    analysis = state.get("analysis_notes", {})
    insights = state.get("insights", {})
    docs = state.get("retrieved_docs", [])

    try:
        report_data = build_full_report(topic, analysis, insights, docs)
        final_report = report_data.get("markdown", "")
        bibtex = report_data.get("bibtex", "")
    except Exception as e:
        logger.error("Report error: %s", e)
        final_report = f"# Error\n\nReport generation failed: {e}"
        bibtex = ""

    return {
        "final_report": final_report,
        "bibtex": bibtex,
        "agent_logs": [{"agent": "report_builder", "message": f"Full Report: {len(final_report)} chars"}],
    }


def node_summarizer(state: GraphState) -> dict:
    """Generate executive summary from full report."""
    if _progress_callback:
        _progress_callback("📋 Generating summary...", 0.95)
    
    full_report = state.get("final_report", "")
    if not full_report:
        return {"summary": "No report available."}
        
    summary = generate_summary(full_report)
    
    return {
        "summary": summary,
        "agent_logs": [{"agent": "summarizer", "message": f"Summary: {len(summary)} chars"}],
    }


def node_error(state: GraphState) -> dict:
    """Terminal error node — no sources found."""
    return {
        "final_report": (
            "# Error\n\n"
            "No sources could be retrieved for this query. "
            "Please try a different topic or check your API keys."
        ),
        "agent_logs": [{"agent": "error_handler", "message": "No sources retrieved"}],
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_retrieve(state: GraphState) -> str:
    """If no docs, route to error; else continue to analyzer."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        logger.warning("No documents retrieved — routing to error handler.")
        return "error_handler"
    logger.info("Retrieved %d docs — routing to analyzer.", len(docs))
    return "analyzer"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def _build_graph():
    """Construct the LangGraph research pipeline.
    
    Structure:
    Decomposer -> Retriever -> Analyzer -> InsightGenerator -> ReportBuilder -> Summarizer -> END
    """
    graph = StateGraph(GraphState)

    # 1. Add Nodes
    graph.add_node("decomposer", node_decompose)
    graph.add_node("retriever", node_retrieve)
    graph.add_node("analyzer", node_analyze)
    graph.add_node("insight_generator", node_insights)
    graph.add_node("report_builder", node_report_builder)
    graph.add_node("summarizer", node_summarizer)
    graph.add_node("error_handler", node_error)

    # 2. Add Edges (Strict Sequence)
    # Entry -> Decomposer
    graph.set_entry_point("decomposer")
    
    # Decomposer -> Retriever
    graph.add_edge("decomposer", "retriever")
    
    # Retriever -> (Analyzer OR Error)
    graph.add_conditional_edges(
        "retriever", 
        route_after_retrieve,
        {
            "analyzer": "analyzer",
            "error_handler": "error_handler"
        }
    )
    
    # Analyzer -> Insight Generator
    graph.add_edge("analyzer", "insight_generator")
    
    # Insight Generator -> Report Builder
    graph.add_edge("insight_generator", "report_builder")
    
    # Report Builder -> Summarizer
    graph.add_edge("report_builder", "summarizer")
    
    # Summarizer -> END
    graph.add_edge("summarizer", END)
    
    # Error Handler -> END
    graph.add_edge("error_handler", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_research(
    topic: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    uploaded_docs: list[dict[str, Any]] | None = None,
) -> ResearchState:
    """Run the full multi-agent research pipeline.

    Args:
        topic: Research topic string.
        progress_callback: Optional (label, fraction) callback for UI.
        uploaded_docs: Optional pre-parsed uploaded documents.

    Returns:
        ResearchState dataclass with all results.
    """
    global _progress_callback
    _progress_callback = progress_callback

    logger.info("=" * 60)
    logger.info("Starting research: %s", topic)
    logger.info("=" * 60)

    initial_state: GraphState = {
        "topic": topic,
        "sub_queries": [],
        "retrieved_docs": [],
        "uploaded_docs": uploaded_docs or [],
        "analysis_notes": {},
        "insights": {},
        "final_report": "",
        "bibtex": "",
        "errors": [],
        "agent_logs": [],
    }

    start = time.time()
    graph = _build_graph()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    if progress_callback:
        progress_callback("✅ Complete!", 1.0)

    logger.info("Pipeline completed in %.1fs", elapsed)

    return ResearchState(
        topic=topic,
        sub_queries=final_state.get("sub_queries", []),
        retrieved_docs=final_state.get("retrieved_docs", []),
        analysis_notes=final_state.get("analysis_notes", {}),
        insights=final_state.get("insights", {}),
        summary=final_state.get("summary", ""),
        final_report=final_state.get("final_report", ""),
        bibtex=final_state.get("bibtex", ""),
        errors=final_state.get("errors", []),
        agent_logs=final_state.get("agent_logs", []),
    )


# ---------------------------------------------------------------------------
# Chat Pipeline (New)
# ---------------------------------------------------------------------------

from agents.chat_agent import classify_intent, answer_from_report, synthesize_answer, check_relevance
from agents.followup_agent import suggest_followups
from utils.document_parser import parse_uploaded_file


class ChatState(TypedDict, total=False):
    """State for the chat pipeline."""
    message: str
    topic: str
    report_context: str
    source_summaries: list[dict]
    chat_history: list[dict]
    uploaded_files: list[Any]  # Streamlit UploadedFile objects
    
    # Internal
    intent: str
    new_docs: list[dict[str, Any]]
    new_analysis: dict[str, Any]
    
    # Output
    response: str
    suggestions: list[str]
    agent_logs: Annotated[list[dict[str, str]], operator.add]


def node_classify(state: ChatState) -> dict:
    # ... (classify node remains same) ...
    """Classify user intent."""
    msg = state.get("message", "")
    topic = state.get("topic", "")
    has_files = bool(state.get("uploaded_files"))
    
    intent = classify_intent(msg, topic, has_files)
    return {
        "intent": intent,
        "agent_logs": [{"agent": "chat_classifier", "message": f"Classified: {intent}"}],
    }

def node_chat_answer(state: ChatState) -> dict:
    # ... (answer node remains same) ...
    """Answer from existing context (cheap)."""
    msg = state.get("message", "")
    report = state.get("report_context", "")
    summaries = state.get("source_summaries", [])
    
    answer = answer_from_report(msg, report, summaries)
    return {
        "response": answer,
        "agent_logs": [{"agent": "chat_agent", "message": "Answered from context"}],
    }

def node_chat_search(state: ChatState) -> dict:
    # ... (search node remains same) ...
    """Retrieve new info from web."""
    msg = state.get("message", "")
    # Treat message as query
    docs = retrieve_sources([msg])
    return {
        "new_docs": docs,
        "agent_logs": [{"agent": "retriever", "message": f"Found {len(docs)} new sources"}],
    }

def node_chat_parse(state: ChatState) -> dict:
    """Parse uploaded files and check relevance."""
    files = state.get("uploaded_files", [])
    topic = state.get("topic", "")
    docs = []
    logs = []
    
    for f in files:
        try:
            content = f.read()
            f.seek(0)
            parsed = parse_uploaded_file(content, f.name)
            
            # Relevance Check
            if check_relevance(topic, parsed.get("content", "")):
                docs.append(parsed)
                logs.append({"agent": "document_parser", "message": f"Parsed relevant file: {f.name}"})
            else:
                logs.append({"agent": "guardrails", "message": f"Ignored irrelevant file: {f.name}"})
                
        except Exception as e:
            logger.error("Failed to parse %s: %s", f.name, e)
            logs.append({"agent": "document_parser", "message": f"Failed to parse: {f.name}"})
            
    return {
        "new_docs": docs,
        "agent_logs": logs,
    }

def node_chat_analyze(state: ChatState) -> dict:
    """Analyze new docs (web or uploaded)."""
    docs = state.get("new_docs", [])
    analysis = analyze_sources(docs)
    return {
        "new_analysis": analysis,
        "agent_logs": [{"agent": "analyzer", "message": "Analyzed new findings"}],
    }

def node_chat_synthesize(state: ChatState) -> dict:
    """Synthesize answer from new analysis."""
    msg = state.get("message", "")
    topic = state.get("topic", "")
    analysis = state.get("new_analysis", {})
    
    answer = synthesize_answer(msg, topic, analysis)
    return {
        "response": answer,
        "agent_logs": [{"agent": "chat_agent", "message": "Synthesized new answer"}],
    }

def node_suggest(state: ChatState) -> dict:
    """Generate follow-up suggestions."""
    msg = state.get("message", "")
    resp = state.get("response", "")
    suggestions = suggest_followups(msg, resp)
    return {"suggestions": suggestions}

def route_chat(state: ChatState) -> str:
    """Route based on intent."""
    intent = state.get("intent", "answer_from_context")
    if intent == "needs_web_search":
        return "chat_search"
    elif intent == "needs_attachment_analysis":
        return "chat_parse"
    else:
        return "chat_answer"

def _build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("classify", node_classify)
    graph.add_node("chat_answer", node_chat_answer)
    graph.add_node("chat_search", node_chat_search)
    graph.add_node("chat_parse", node_chat_parse)
    graph.add_node("chat_analyze", node_chat_analyze)
    graph.add_node("chat_synthesize", node_chat_synthesize)
    graph.add_node("suggest", node_suggest)
    
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_chat)
    
    graph.add_edge("chat_answer", "suggest")
    graph.add_edge("chat_search", "chat_analyze")
    graph.add_edge("chat_parse", "chat_analyze")
    graph.add_edge("chat_analyze", "chat_synthesize")
    graph.add_edge("chat_synthesize", "suggest")
    graph.add_edge("suggest", END)
    
    return graph.compile()

@dataclass
class ChatResponse:
    """Response from the chat pipeline."""
    response: str
    suggestions: list[str]
    intent: str
    agent_logs: list[dict[str, str]]

def run_chat(
    message: str,
    topic: str,
    report_context: str,
    source_summaries: list[dict],
    chat_history: list[dict],
    uploaded_files: list[Any] = None,
) -> ChatResponse:
    """Run the chat pipeline."""
    logger.info("Chat: %s (files: %d)", message[:50], len(uploaded_files or []))
    
    initial: ChatState = {
        "message": message,
        "topic": topic,
        "report_context": report_context,
        "source_summaries": source_summaries,
        "chat_history": chat_history,
        "uploaded_files": uploaded_files or [],
        "agent_logs": [],
    }
    
    graph = _build_chat_graph()
    final = graph.invoke(initial)
    
    return ChatResponse(
        response=final.get("response", "I'm sorry, I couldn't generate a response."),
        suggestions=final.get("suggestions", []),
        intent=final.get("intent", "unknown"),
        agent_logs=final.get("agent_logs", []),
    )
