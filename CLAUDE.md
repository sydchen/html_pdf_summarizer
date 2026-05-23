# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PDF & Web Summarizer application that uses local AI models via Ollama to generate summaries of PDF documents and web articles. The application features a Streamlit frontend with real-time processing feedback and supports both file uploads and URL inputs.

## Architecture

### Core Components

- **Frontend (`frontend.py`)**: Streamlit-based web interface handling user interactions, file uploads, and URL inputs with streaming status updates
- **Routing (`summarizer_service.py`, `source_detection.py`)**: Shared entrypoint and source classification for Streamlit and CLI flows
- **PDF Processing (`pdf_summarizer.py`)**: PDF text extraction using PyPDF2 with intelligent chunking for large documents
- **Web Content Processing (`html_summarizer.py`)**: Handles both HTML articles and PDF URLs with content type detection and BeautifulSoup parsing
- **YouTube/Transcript Processing (`youtube_summarizer.py`, `transcript_llm.py`)**: Handles YouTube URLs, local videos, SRT/TXT transcripts, Whisper transcription, and transcript summaries
- **LLM Interface (`llm.py`)**: Wrapper for Ollama integration with conversation management and system prompts optimized for Chinese summarization

### Data Flow

1. User uploads PDF or enters URL via Streamlit interface
2. Shared source detection determines processing path (PDF, HTML, YouTube, local video, transcript)
3. Text extraction and intelligent chunking based on token limits
4. Iterative summarization using Ollama's Gemma 3 model
5. Multi-stage summary reduction for long documents
6. Real-time streaming of results to frontend

### Key Design Patterns

- **Async/Sync Hybrid**: Uses asyncio for web requests while maintaining sync interface for Streamlit compatibility
- **Progressive Summarization**: Large documents are chunked, summarized individually, then recursively merged
- **Error Recovery**: Multiple fallback strategies for web content fetching and processing
- **Token Management**: Dynamic token limit adjustment based on content length and document type

## Development Commands

### Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Download required model
ollama pull gemma3:4b
```

### Running the Application

```bash
# Start the Streamlit frontend
streamlit run frontend.py

# Test individual components
python3 -m unittest discover -s tests -q
python3 pdf_summarizer.py
python3 html_summarizer.py
python3 llm.py
```

### Environment Configuration

The application expects a MODEL environment variable or uses `gemma3:4b` as default. Copy `.env.example` to `.env` and adjust local paths:

```
MODEL=gemma3:4b
```

## Technical Notes

- **PDF Processing**: Uses PyPDF2 with text cleaning and line merging heuristics for better readability
- **Web Scraping**: Targets main content areas using BeautifulSoup with fallback content detection
- **Token Counting**: Uses lightweight character-based estimation before LLM processing
- **Chunking Strategy**: Respects paragraph boundaries and implements overlapping for context preservation
- **Streaming**: Real-time progress updates through Streamlit's status components

## Dependencies

Key dependencies include:
- `streamlit` - Web interface framework
- `ollama` - Local LLM integration
- `PyPDF2` - PDF text extraction
- `beautifulsoup4` - HTML parsing
- `httpx` - Async HTTP client
- `prefect` - YouTube/video workflow task wrappers

## File Import Dependencies

- `frontend.py` uses `SummarizerService`
- `summarizer_service.py` routes to `pdf_summarizer.py`, `html_summarizer.py`, or `youtube_summarizer.py`
- `pdf_summarizer.py` and `html_summarizer.py` depend on `llm.py`
- `youtube_summarizer.py` depends on `transcript_llm.py` for transcript-specific summaries
