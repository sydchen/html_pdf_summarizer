import re
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse


class SourceType(str, Enum):
    """Supported input categories for the summarizer service."""

    HTML_URL = "html_url"
    PDF_URL = "pdf_url"
    YOUTUBE_URL = "youtube_url"
    PDF_FILE = "pdf_file"
    VIDEO_FILE = "video_file"
    TRANSCRIPT_FILE = "transcript_file"
    UNKNOWN = "unknown"


YOUTUBE_PATTERNS = (
    re.compile(r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"^(https?://)?(www\.)?youtu\.be/[\w-]+"),
    re.compile(r"^(https?://)?(www\.)?youtube\.com/embed/[\w-]+"),
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
TRANSCRIPT_SUFFIXES = {".srt", ".txt"}
PDF_SUFFIX = ".pdf"


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_youtube_url(value: str) -> bool:
    return any(pattern.match(value) for pattern in YOUTUBE_PATTERNS)


def is_pdf_url(value: str) -> bool:
    parsed = urlparse(value)
    return is_url(value) and parsed.path.lower().endswith(PDF_SUFFIX)


def is_transcript_file(value: str) -> bool:
    if is_url(value):
        return False
    return Path(value).suffix.lower() in TRANSCRIPT_SUFFIXES


def is_video_file(value: str) -> bool:
    if is_url(value):
        return False
    return Path(value).suffix.lower() in VIDEO_SUFFIXES


def detect_source_type(value: str) -> SourceType:
    """Classify a string input without touching the network or filesystem."""
    if is_youtube_url(value):
        return SourceType.YOUTUBE_URL
    if is_pdf_url(value):
        return SourceType.PDF_URL
    if is_url(value):
        return SourceType.HTML_URL

    suffix = Path(value).suffix.lower()
    if suffix == PDF_SUFFIX:
        return SourceType.PDF_FILE
    if suffix in VIDEO_SUFFIXES:
        return SourceType.VIDEO_FILE
    if suffix in TRANSCRIPT_SUFFIXES:
        return SourceType.TRANSCRIPT_FILE
    return SourceType.UNKNOWN
