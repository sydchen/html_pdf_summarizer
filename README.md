# PDF & Web Summarizer

基於 Ollama 本地 AI 模型的文件摘要工具，支援 PDF、網頁文章、YouTube 影片、本地影片與逐字稿摘要。

## 主要功能

- **多格式支援**: PDF 上傳、網頁/PDF URL、YouTube URL、本地影片、SRT/TXT 逐字稿
- **本地AI處理**: 使用 Ollama 模型，保護隱私安全
- **即時顯示**: 摘要生成過程即時可見

## Tech Stack

| Component          |                                                                            |
| ------------------ | -------------------------------------------------------------------------- |
| Frontend       | [Streamlit](https://streamlit.io/)                                         |
| LLM Platform   | [Ollama](https://ollama.com/)                                              |
| LLM Model      | [Google Gemma 3](https://developers.googleblog.com/en/introducing-gemma3/) |
| PDF Processing | [PyPDF2](https://pypi.org/project/PyPDF2/)                                 |
| Video Pipeline | yt-dlp, ffmpeg, whisper.cpp                                                |

---

## Supported Inputs

| Input | Where to use it |
| --- | --- |
| Uploaded PDF | Streamlit frontend |
| Web article URL | Streamlit frontend or `html_summarizer.py` |
| PDF URL | Streamlit frontend or `html_summarizer.py` |
| YouTube URL | Streamlit frontend, `html_summarizer.py`, or `youtube_summarizer.py` |
| Local video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`) | `youtube_summarizer.py` |
| Transcript file (`.srt`, `.txt`) | `youtube_summarizer.py` or `transcript_llm.py` |

## Scripts

| Script | Purpose |
| --- | --- |
| `frontend.py` | Streamlit UI for PDF upload, URL summary, and YouTube summary |
| `html_summarizer.py` | CLI for web article, PDF URL, or YouTube URL summaries |
| `pdf_summarizer.py` | PDF extraction and summarization helper used by the service/UI |
| `youtube_summarizer.py` | CLI for YouTube URLs, local videos, and transcript files |
| `clean_transcript.py` | Utility for converting SRT files into cleaned plain text |
| `transcript_llm.py` | Transcript-focused summarizer and chunking test CLI |
| `summarizer_service.py` | Shared routing entrypoint used by the frontend |
| `source_detection.py` | Input type detection helpers |
| `chunking.py` | Shared document/transcript chunking helpers |

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/sydchen/html_pdf_summarizer.git
cd html_pdf_summarizer
```

### Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Install Ollama and Gemma 3 LLM

Install Ollama - MacOS/Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama and download Gemma 3 Model

```bash
ollama serve & ollama pull gemma3:4b
```

### Configure Environment

```bash
cp .env.example .env
```

Update `.env` if you use a different Ollama model, Whisper model path, or output directory. `.env` is for local secrets and machine-specific paths only; do not commit it.

## Usage

### Start the Frontend

```bash
streamlit run frontend.py
```

### Summarize a Web Page, PDF URL, or YouTube URL

```bash
python3 html_summarizer.py "https://example.com/article"
python3 html_summarizer.py "https://example.com/paper.pdf" --task-type academic_paper
python3 html_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Available `--task-type` values:

- `short_summary`
- `long_summary`
- `detailed_analysis`
- `academic_paper`

### Summarize YouTube, Local Video, or Transcript Files

```bash
python3 youtube_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
python3 youtube_summarizer.py ./video.mp4 --lang zh
python3 youtube_summarizer.py ./lecture.srt
python3 youtube_summarizer.py ./notes.txt
```

For local video files, `--lang` can be `auto`, `zh`, `ja`, `en`, `ko`, or another Whisper-supported language code.

### Clean an SRT Transcript

```bash
python3 clean_transcript.py lecture.srt
python3 clean_transcript.py lecture.srt lecture_clean.txt
python3 clean_transcript.py lecture.srt --stdout
```

By default, `clean_transcript.py lecture.srt` writes `lecture_clean.txt` next to the input file.

### Summarize a Transcript Directly

```bash
python3 transcript_llm.py ./lecture.srt
python3 transcript_llm.py ./lecture.txt --chunk-size 8000 --overlap-words 200
```

## External Tools for Video Summaries

YouTube and local video summaries require these command-line tools in addition to Python dependencies:

- `yt-dlp` for downloading YouTube videos
- `ffmpeg` for extracting/converting audio
- `whisper.cpp` for audio transcription

Configure `WHISPER_MODEL_PATH` and `WHISPER_BINARY_PATH` in `.env` before running video transcription.

### Run Tests

```bash
python3 -m unittest discover -s tests -q
```
