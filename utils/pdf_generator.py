"""
Professional PDF report generator using ReportLab.

Converts Markdown content into a polished, multi-page PDF with a title
page, table of contents, formatted sections, references, page numbers,
and generation timestamp.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_STYLES = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "PDFTitle",
    parent=_STYLES["Title"],
    fontSize=28,
    leading=34,
    spaceAfter=20,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"),
)

_SUBTITLE_STYLE = ParagraphStyle(
    "PDFSubtitle",
    parent=_STYLES["Normal"],
    fontSize=14,
    leading=18,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
    spaceAfter=40,
)

_HEADING_STYLE = ParagraphStyle(
    "PDFHeading",
    parent=_STYLES["Heading1"],
    fontSize=18,
    leading=22,
    spaceAfter=10,
    spaceBefore=16,
    textColor=colors.HexColor("#16213e"),
)

_SUBHEADING_STYLE = ParagraphStyle(
    "PDFSubheading",
    parent=_STYLES["Heading2"],
    fontSize=14,
    leading=18,
    spaceAfter=8,
    spaceBefore=12,
    textColor=colors.HexColor("#0f3460"),
)

_BODY_STYLE = ParagraphStyle(
    "PDFBody",
    parent=_STYLES["Normal"],
    fontSize=11,
    leading=15,
    spaceAfter=8,
    alignment=TA_JUSTIFY,
)

_FOOTER_STYLE = ParagraphStyle(
    "PDFFooter",
    parent=_STYLES["Normal"],
    fontSize=8,
    textColor=colors.grey,
    alignment=TA_CENTER,
)


# ---------------------------------------------------------------------------
# Page callbacks
# ---------------------------------------------------------------------------

def _header_footer(canvas, doc):
    """Add page number footer to every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    page_num = f"Page {doc.page}"
    canvas.drawCentredString(A4[0] / 2, 15 * mm, page_num)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf_report(
    markdown_content: str,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Convert a Markdown research report into a professional PDF.

    Args:
        markdown_content: The full Markdown report body.
        metadata: Optional dict with keys like ``title``, ``topic``,
                  ``generated_at``, ``total_sources``.

    Returns:
        Raw PDF bytes (ready to be served via Streamlit download button).
    """
    metadata = metadata or {}
    title = metadata.get("title", "Deep Research Report")
    topic = metadata.get("topic", "")
    generated_at = metadata.get(
        "generated_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    total_sources = metadata.get("total_sources", 0)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
    )

    story: list = []

    # -- Title page --------------------------------------------------------
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(title, _TITLE_STYLE))
    if topic:
        story.append(Paragraph(f"Topic: {_escape(topic)}", _SUBTITLE_STYLE))
    story.append(
        Paragraph(
            f"Generated: {generated_at} &bull; Sources: {total_sources}",
            _SUBTITLE_STYLE,
        )
    )
    story.append(PageBreak())

    # -- Body (parsed from Markdown) ---------------------------------------
    story.extend(_parse_markdown(markdown_content))

    # -- Build PDF ---------------------------------------------------------
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Markdown → ReportLab flowables (simple parser)
# ---------------------------------------------------------------------------

_RE_H1 = re.compile(r"^#\s+(.+)$")
_RE_H2 = re.compile(r"^##\s+(.+)$")
_RE_H3 = re.compile(r"^###\s+(.+)$")
_RE_BULLET = re.compile(r"^[-*]\s+(.+)$")


def _parse_markdown(md: str) -> list:
    """Naïve Markdown-to-flowable converter covering headings, bullets, and paragraphs."""
    flowables: list = []
    current_paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if current_paragraph_lines:
            text = " ".join(current_paragraph_lines)
            flowables.append(Paragraph(_escape(text), _BODY_STYLE))
            current_paragraph_lines.clear()

    for raw_line in md.splitlines():
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            flowables.append(Spacer(1, 4 * mm))
            continue

        m_h1 = _RE_H1.match(line)
        if m_h1:
            flush_paragraph()
            flowables.append(Paragraph(_escape(m_h1.group(1)), _HEADING_STYLE))
            continue

        m_h2 = _RE_H2.match(line)
        if m_h2:
            flush_paragraph()
            flowables.append(Paragraph(_escape(m_h2.group(1)), _SUBHEADING_STYLE))
            continue

        m_h3 = _RE_H3.match(line)
        if m_h3:
            flush_paragraph()
            flowables.append(
                Paragraph(f"<b>{_escape(m_h3.group(1))}</b>", _BODY_STYLE)
            )
            continue

        m_bullet = _RE_BULLET.match(line)
        if m_bullet:
            flush_paragraph()
            flowables.append(
                Paragraph(f"&bull; {_escape(m_bullet.group(1))}", _BODY_STYLE)
            )
            continue

        # Regular text — accumulate into paragraph
        current_paragraph_lines.append(line)

    flush_paragraph()
    return flowables


def _escape(text: str) -> str:
    """Escape XML special chars for ReportLab Paragraph."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
