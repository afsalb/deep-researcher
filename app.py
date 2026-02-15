"""
Deep Researcher — Streamlit Application.

Glassmorphic Material 3 UI:
- Translucent glass cards with blur effects
- Animated mesh gradient background
- Responsive CSS Grid layouts for metrics & quick starts
- High-fidelity typography (Inter/JetBrains Mono)
- Universal document upload & multi-agent orchestration
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

import streamlit as st

from agents.orchestrator import run_research, run_chat, ResearchState
from config.models import QUERY_MAX_LENGTH
from utils.cost_tracker import CostTracker
from utils.document_parser import parse_uploaded_file
from utils.pdf_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Glassmorphic CSS — Responsive & Theme-Aware
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ========== IMPORTS ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ========== DYNAMIC VARIABLES ========== */
:root {
    --primary: var(--primary-color);
    --bg-color: var(--background-color);
    --text-color: var(--text-color);
    
    /* Dynamic Glass System */
    --glass-surface: color-mix(in srgb, var(--bg-color), transparent 20%);
    --glass-border: color-mix(in srgb, var(--text-color), transparent 88%);
    --glass-shadow: 0 4px 20px 0 color-mix(in srgb, var(--text-color), transparent 95%);
    
    --text-main: var(--text-color);
    --text-muted: color-mix(in srgb, var(--text-color), transparent 45%);
    
    --radius-lg: 20px;
    --radius-md: 12px;
    --radius-sm: 8px;
}

/* ========== GLOBAL RESET ========== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}

/* ========== MINIMALIST BACKGROUND (SUBTLE) ========== */
.stApp {
    background-image: 
        radial-gradient(at 0% 0%, color-mix(in srgb, var(--primary), transparent 96%) 0px, transparent 50%),
        radial-gradient(at 100% 0%, color-mix(in srgb, var(--primary), transparent 97%) 0px, transparent 50%),
        radial-gradient(at 100% 100%, color-mix(in srgb, var(--primary), transparent 98%) 0px, transparent 50%);
    background-attachment: fixed;
    background-size: cover;
    background-color: var(--bg-color);
}

/* ========== CHAT STYLES ========== */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    padding-top: 1rem;
    margin-bottom: 2rem;
}

.chat-row {
    display: flex;
    gap: 1rem;
    align-items: flex-end;
    width: 100%;
}

.chat-row.user-row {
    flex-direction: row-reverse;
}

.chat-avatar {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    background: var(--glass-surface);
    border: 1px solid var(--glass-border);
}

/* BUBBLES */
.chat-bubble {
    padding: 1rem 1.4rem;
    line-height: 1.6;
    font-size: 0.95rem;
    max-width: 75%;
    position: relative;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    transition: transform 0.2s;
}

.chat-bubble:hover {
    transform: translateY(-1px);
}

/* User Bubble: Modern Gradient & Sharp Corner */
.chat-user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border-radius: 20px 20px 4px 20px;
    border: none;
}

.chat-user-bubble * {
    color: white !important;
}

.chat-user-bubble a {
    color: white !important;
    text-decoration: underline;
}

.chat-user-bubble p {
    color: white !important;
    margin: 0;
}

/* AI Bubble: Glassmorphic & Sharp Corner */
.chat-ai-bubble {
    background: var(--glass-surface);
    border: 1px solid var(--glass-border);
    color: var(--text-main) !important;
    border-radius: 20px 20px 20px 4px;
    backdrop-filter: blur(12px);
}

.chat-ai-bubble * {
    color: var(--text-main) !important;
}

.chat-ai-bubble p {
    color: var(--text-main) !important;
    margin: 0;
}

/* BADGES */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.70rem;
    padding: 4px 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 20px;
    margin-top: 0.8rem;
    margin-right: 0.5rem;
    color: inherit;
    opacity: 0.9;
    font-weight: 500;
}

.chat-user-bubble .source-badge {
    background: rgba(255,255,255,0.2);
    color: white !important;
}

.suggestion-chip {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
}

.suggestion-chip:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--primary);
    transform: translateY(-1px);
}

/* ========== HEADER ========== */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ========== HIGH CONTRAST INPUTS ========== */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
    background-color: color-mix(in srgb, var(--bg-color), transparent 5%) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-main) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.95rem !important;
    box-shadow: none !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stTextInput input:hover, .stTextArea textarea:hover, .stSelectbox div[data-baseweb="select"]:hover {
    border-color: var(--text-muted) !important;
    background-color: var(--bg-color) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
    border-color: var(--primary) !important;
    background-color: var(--bg-color) !important;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary), transparent 75%) !important;
}

/* ========== METRICS & CARDS ========== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
    margin-bottom: 2rem;
}

.glass-metric {
    background: var(--glass-surface);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1rem;
    box-shadow: var(--glass-shadow);
}

.metric-label {
    font-size: 0.70rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
    margin-top: 0.2rem;
}

/* ========== UPLOAD ZONE ========== */
.glass-upload {
    background: color-mix(in srgb, var(--bg-color), transparent 40%);
    border: 2px dashed var(--glass-border);
    border-radius: var(--radius-md);
    padding: 2rem;
    text-align: center;
    transition: all 0.2s;
    cursor: pointer;
}

.glass-upload:hover {
    border-color: var(--primary);
    background: color-mix(in srgb, var(--bg-color), transparent 10%);
}

/* ========== BUTTONS ========== */
div[data-testid="stButton"] button {
    border-radius: var(--radius-md) !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 500 !important;
    border: 1px solid var(--glass-border) !important;
    background: var(--glass-surface) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    transition: all 0.2s !important;
}

div[data-testid="stButton"] button:hover {
    border-color: var(--primary) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px color-mix(in srgb, var(--text-color), transparent 90%) !important;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #5568d3, #6a3f8f) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* ========== LOG ========== */
.agent-log {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 400px;
    overflow-y: auto;
}

.log-pill {
    background: color-mix(in srgb, var(--bg-color), transparent 10%);
    border: 1px solid var(--glass-border);
    padding: 0.6rem 0.8rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    display: flex;
    gap: 0.6rem;
    align-items: center;
}

.agent-badge {
    background: var(--text-muted);
    color: var(--bg-color);
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 1rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    color: var(--text-muted);
    font-weight: 500;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary);
    border-bottom: 2px solid var(--primary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False
if "research_count" not in st.session_state:
    st.session_state.research_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestions" not in st.session_state:
    try:
        from agents.followup_agent import suggest_followups
        st.session_state.suggestions = ["Tell me more about the key findings.", "What are the limitations?", "Compare this with other approaches."]
    except ImportError:
         st.session_state.suggestions = []

# ---------------------------------------------------------------------------
# Sidebar (Metrics & Log)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 Deep Researcher")
    st.markdown("<div style='margin-bottom: 1.5rem; font-size: 0.9rem; color: var(--text-muted);'>AI-powered multi-hop investigation engine</div>", unsafe_allow_html=True)

    # -- Metrics Grid (Responsive) --
    tracker = CostTracker()
    stats = tracker.get_session_stats()
    
    metrics_html = f"""
    <div class="metrics-grid">
        <div class="glass-metric">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${stats["total_cost"]:.4f}</div>
        </div>
        <div class="glass-metric">
            <div class="metric-label">Tokens</div>
            <div class="metric-value">{(stats["total_input_tokens"] + stats["total_output_tokens"]) / 1000:.1f}k</div>
        </div>
        <div class="glass-metric">
            <div class="metric-label">Calls</div>
            <div class="metric-value">{stats["total_calls"]}</div>
        </div>
        <div class="glass-metric">
            <div class="metric-label">Researches</div>
            <div class="metric-value">{st.session_state.research_count}</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown("---")
    
    # -- Agent Activity Log --
    st.markdown("#### ⚡ Live Agent Activity")
    result: ResearchState | None = st.session_state.result
    
    if result and result.agent_logs:
        log_html = '<div class="agent-log">'
        for entry in reversed(result.agent_logs[-15:]):
            agent = html.escape(str(entry.get("agent", "System")))
            msg = html.escape(str(entry.get("message", "")))
            log_html += (
                f'<div class="log-pill">'
                f'<span class="agent-badge">{agent}</span>'
                f'<span>{msg}</span>'
                f'</div>'
            )
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("Ready to research. Activity will appear here.")

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-container">'
    '<div class="hero-title">Deep Research Engine</div>'
    '<div class="hero-subtitle">Ask complex questions. Get comprehensive reports backed by academic papers, web search, and your documents.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# -- Quick Starters --
start_topics = [
    "Impact of LLMs on scientific discovery",
    "CRISPR gene therapy ethics 2024",
    "Solid-state battery breakthroughs",
    "AI in medical diagnostics",
    "Quantum computing limitations",
    "Blockchain supply chain transparency"
]

st.markdown("##### ⚡ Quick Start")
cols = st.columns(3)
for i, topic in enumerate(start_topics):
    if cols[i % 3].button(topic, key=f"quick_{i}"):
        st.session_state["query_input"] = topic

# -- Upload Zone --
st.markdown("##### 📎 Add Context (Optional)")
with st.container():
    st.markdown(
        '<div class="glass-upload">Drag and drop PDF, DOCX, Images, or Audio files here</div>', 
        unsafe_allow_html=True
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "mp3", "wav", "mp4"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# -- Search Input --
st.markdown("##### 🔍 Research Query")
query = st.text_area(
    "Research Topic",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g. detailed analysis of graphene manufacturing costs...",
    label_visibility="collapsed"
)

# Research button with full width
run_btn = st.button(
    "🚀 Start Deep Research", 
    type="primary", 
    disabled=st.session_state.running,
    use_container_width=True
)

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

if run_btn and query:
    st.session_state.running = True
    tracker.reset()
    
    # Parse uploads
    parsed_uploads = []
    if uploaded_files:
        with st.spinner("📄 Analyzing uploaded documents..."):
            for f in uploaded_files:
                parsed_uploads.append(parse_uploaded_file(f.getvalue(), f.name))
    
    # Progress callback
    progress_bar = st.progress(0, text="Initializing agents...")
    
    def _update_progress(label: str, pct: float):
        progress_bar.progress(pct, text=label)
        
    try:
        result_state = run_research(
            query,
            progress_callback=_update_progress,
            uploaded_docs=parsed_uploads
        )
        st.session_state.result = result_state
        st.session_state.research_count += 1
        st.rerun()
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
    finally:
        st.session_state.running = False

# ---------------------------------------------------------------------------
# Results View
# ---------------------------------------------------------------------------

if st.session_state.result:
    res = st.session_state.result
    st.markdown("---")
    st.markdown(f"### 🏁 Results: {res.topic}")
    
    tabs = st.tabs(["📋 Summary", "💡 Insights", "📄 Full Report", "📚 Sources"])
    
    with tabs[0]:
        if hasattr(res, 'summary') and res.summary:
            st.markdown(res.summary)
        else:
            st.markdown(res.final_report.split("## Key Findings")[0])
        
    with tabs[1]:
        if res.insights:
            for k, v in res.insights.items():
                if v:
                    with st.expander(k.title().replace("_", " "), expanded=True):
                        st.markdown("\n".join([f"- {i}" for i in v]) if isinstance(v, list) else str(v))
                        
    with tabs[2]:
        st.markdown(res.final_report)
        
    with tabs[3]:
        for doc in res.retrieved_docs:
            icon = "📄" if doc.get("source_type") == "uploaded" else "🌐"
            with st.expander(f"{icon} {doc.get('title')}"):
                st.write(doc.get("content"))
                st.caption(f"Source: {doc.get('url')}")
    
    # Downloads
    st.markdown("### 💾 Exports")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download MarkDown", res.final_report, "report.md")
    try:
        pdf_bytes = generate_pdf_report(res.final_report, {"topic": res.topic})
        d2.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
    except Exception as exc:
        d2.error(f"PDF Error: {exc}")
    
    if res.bibtex:
        d3.download_button("Download BibTeX", res.bibtex, "references.bib", "text/plain")

# ---------------------------------------------------------------------------
# Chat Interface (Fixed)
# ---------------------------------------------------------------------------

if st.session_state.result:
    st.markdown("---")
    st.markdown("### 💬 Deep Chat")
    
    # -- Chat Container --
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Render chat history
    for i, chat in enumerate(st.session_state.chat_history):
        role = chat["role"]
        content = chat["content"]
        
        if role == "user":
            avatar = "👤"
            row_class = "user-row"
            bubble_class = "chat-user-bubble"
        else:
            avatar = "🤖"
            row_class = "ai-row"
            bubble_class = "chat-ai-bubble"
        
        # Build badges HTML
        badges_html = ""
        if role == "assistant" and "agent_logs" in chat:
            for log in chat["agent_logs"]:
                if "Found" in log.get("message", ""):
                    badges_html += '<span class="source-badge">🌐 Search</span>'
                if "Parsed" in log.get("message", ""):
                    badges_html += '<span class="source-badge">📄 Doc</span>'
                if "Ignored" in log.get("message", ""):
                    badges_html += '<span class="source-badge" style="border-color:#ff4444; color:#ff8888">🛡️ Blocked</span>'

        # Format content - always escape for safety, then convert newlines
        formatted_content = html.escape(content).replace('\n', '<br>')
        
        # Add badges after content with spacing
        if badges_html:
            formatted_content += '<br>' + badges_html
        
        # Build HTML in single line to prevent Streamlit code block rendering
        html_content = f'<div class="chat-row {row_class}"><div class="chat-avatar">{avatar}</div><div class="chat-bubble {bubble_class}">{formatted_content}</div></div>'
        
        st.markdown(html_content, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Suggestions --
    st.markdown("")
    
    def handle_suggestion(sugg_text):
        st.session_state.chat_trigger = sugg_text

    if "suggestions" in st.session_state and st.session_state.suggestions:
        cols = st.columns(len(st.session_state.suggestions))
        for i, sugg in enumerate(st.session_state.suggestions):
            if cols[i].button(sugg, key=f"sugg_btn_{len(st.session_state.chat_history)}_{i}"):
                handle_suggestion(sugg)

    # -- Input Area --
    st.markdown("##### Ask a follow-up")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col_text, col_file = st.columns([4, 1])
        
        initial_val = st.session_state.get("chat_trigger", "")
        
        user_msg = col_text.text_input(
            "Message", 
            value=initial_val,
            placeholder="Ask about specific findings...",
            key="user_input_widget"
        )
        
        new_files = col_file.file_uploader(
            "Attach", 
            type=["pdf", "docx", "txt", "md"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Send")

    # Clear trigger after render
    if "chat_trigger" in st.session_state:
        del st.session_state.chat_trigger
        
    if submitted:
        if not user_msg and not new_files:
            st.warning("Please type a message or upload a file.")
        else:
            # Add user message
            display_msg = user_msg
            if new_files:
                display_msg += f" [Attached {len(new_files)} files]"
                
            st.session_state.chat_history.append({"role": "user", "content": display_msg})
            
            # Run chat pipeline
            with st.spinner("Thinking..."):
                res = st.session_state.result
                chat_response = run_chat(
                    message=user_msg,
                    topic=res.topic,
                    report_context=res.final_report,
                    source_summaries=res.retrieved_docs,
                    chat_history=st.session_state.chat_history,
                    uploaded_files=new_files
                )
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": chat_response.response,
                    "agent_logs": chat_response.agent_logs
                })
                
                # Update suggestions
                st.session_state.suggestions = chat_response.suggestions
            
            st.rerun()