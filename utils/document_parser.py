"""
Universal document parser — extracts text/content from any uploaded file.

Supported formats:
- **Text**: TXT, MD (direct read)
- **PDF**: via PyPDF2
- **DOCX**: via python-docx
- **Images**: PNG, JPG, JPEG, WEBP (LLM vision via OpenRouter)
- **Audio**: MP3, WAV, OGG (LLM multimodal via OpenRouter)
- **Video**: MP4, WEBM, AVI (keyframe extraction + LLM vision)

All outputs are normalised to the same dict schema used by Tavily/Arxiv
results so they integrate seamlessly into the research pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from config.models import CONTENT_TRUNCATION_LIMIT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIME / extension mappings
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml"}
_PDF_EXTENSIONS = {".pdf"}
_DOCX_EXTENSIONS = {".docx"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_uploaded_file(
    file_bytes: bytes,
    filename: str,
) -> dict[str, Any]:
    """Parse an uploaded file and return a normalised source dict.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used for extension detection).

    Returns:
        Dict with keys: ``title``, ``content``, ``url``, ``source_type``,
        ``published_date``, ``relevance_score``, ``file_type``.
    """
    ext = os.path.splitext(filename)[1].lower()
    logger.info("Parsing uploaded file: %s (ext=%s, size=%d bytes)", filename, ext, len(file_bytes))

    try:
        if ext in _TEXT_EXTENSIONS:
            content = _parse_text(file_bytes)
        elif ext in _PDF_EXTENSIONS:
            content = _parse_pdf(file_bytes)
        elif ext in _DOCX_EXTENSIONS:
            content = _parse_docx(file_bytes)
        elif ext in _IMAGE_EXTENSIONS:
            content = _parse_image(file_bytes, ext)
        elif ext in _AUDIO_EXTENSIONS:
            content = _parse_audio(file_bytes, ext)
        elif ext in _VIDEO_EXTENSIONS:
            content = _parse_video(file_bytes, ext)
        else:
            # Best-effort: try as text
            content = _parse_text(file_bytes)
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", filename, exc)
        content = f"[Upload parsing failed: {exc}]"

    return {
        "title": filename,
        "content": content[:CONTENT_TRUNCATION_LIMIT],
        "url": f"uploaded://{filename}",
        "source_type": "uploaded",
        "published_date": "",
        "relevance_score": 1.0,  # user-uploaded = highest relevance
        "file_type": ext.lstrip("."),
    }


# ---------------------------------------------------------------------------
# Text / Markdown
# ---------------------------------------------------------------------------


def _parse_text(data: bytes) -> str:
    """Decode raw bytes as UTF-8 text."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


# ---------------------------------------------------------------------------
# PDF (PyPDF2)
# ---------------------------------------------------------------------------


def _parse_pdf(data: bytes) -> str:
    """Extract all text pages from a PDF."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return "[PyPDF2 not installed — cannot parse PDF]"

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages) if pages else "[No extractable text in PDF]"


# ---------------------------------------------------------------------------
# DOCX (python-docx)
# ---------------------------------------------------------------------------


def _parse_docx(data: bytes) -> str:
    """Extract text from a Word document."""
    try:
        from docx import Document
    except ImportError:
        return "[python-docx not installed — cannot parse DOCX]"

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs) if paragraphs else "[No extractable text in DOCX]"


# ---------------------------------------------------------------------------
# Images → LLM Vision
# ---------------------------------------------------------------------------


def _parse_image(data: bytes, ext: str) -> str:
    """Send image to OpenRouter multimodal model for description."""
    mime = _IMAGE_MIME.get(ext, "image/png")
    b64 = base64.b64encode(data).decode("ascii")

    return _llm_multimodal_describe(
        data_uri=f"data:{mime};base64,{b64}",
        media_type="image",
        prompt=(
            "You are a research assistant. Describe this image in detail for "
            "academic research purposes. Include any text, data, charts, "
            "diagrams, or key information visible in the image."
        ),
    )


# ---------------------------------------------------------------------------
# Audio → LLM Multimodal
# ---------------------------------------------------------------------------


def _parse_audio(data: bytes, ext: str) -> str:
    """Send audio to OpenRouter multimodal model for transcription."""
    mime = _AUDIO_MIME.get(ext, "audio/mpeg")
    b64 = base64.b64encode(data).decode("ascii")

    return _llm_multimodal_describe(
        data_uri=f"data:{mime};base64,{b64}",
        media_type="audio",
        prompt=(
            "You are a research assistant. Transcribe and summarise this "
            "audio content. Extract all spoken words, key topics discussed, "
            "and any notable claims or data points mentioned."
        ),
    )


# ---------------------------------------------------------------------------
# Video → Keyframe extraction + LLM Vision
# ---------------------------------------------------------------------------


def _parse_video(data: bytes, ext: str) -> str:
    """Extract keyframes from video and analyse via LLM vision."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return "[opencv-python-headless not installed — cannot parse video]"

    # Write to temp file for OpenCV
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()

    try:
        cap = cv2.VideoCapture(tmp.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return "[Could not read video frames]"

        # Extract 3 evenly spaced keyframes
        frame_indices = [
            int(total_frames * pct) for pct in [0.1, 0.5, 0.9]
        ]
        descriptions: list[str] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Encode frame as JPEG
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            desc = _llm_multimodal_describe(
                data_uri=f"data:image/jpeg;base64,{b64}",
                media_type="image",
                prompt=(
                    f"You are a research assistant. Describe this video frame "
                    f"(frame {idx + 1}/{total_frames}) in detail. Note any "
                    f"text, slides, diagrams, people, or data visible."
                ),
            )
            descriptions.append(f"[Frame {idx + 1}/{total_frames}] {desc}")
        cap.release()
    finally:
        os.unlink(tmp.name)

    return "\n\n".join(descriptions) if descriptions else "[No frames extracted from video]"


# ---------------------------------------------------------------------------
# Shared LLM multimodal call
# ---------------------------------------------------------------------------


def _llm_multimodal_describe(
    data_uri: str,
    media_type: str,
    prompt: str,
) -> str:
    """Call OpenRouter multimodal model to describe media content.

    Uses Gemini 2.0 Flash (low-cost, multimodal-capable) via the
    existing LLMClient infrastructure.
    """
    from utils.llm_client import LLMClient

    client = LLMClient()

    if media_type == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]
    else:
        # For audio — send as text description request with base64 context
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ]

    try:
        result = client.get_chat_completion(
            messages=messages,
            model_tier="low_cost",
            task_name=f"document_parser_{media_type}",
        )
        return result
    except Exception as exc:
        logger.error("Multimodal LLM call failed for %s: %s", media_type, exc)
        return f"[Could not analyse {media_type}: {exc}]"
