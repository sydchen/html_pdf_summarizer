from collections.abc import Callable, Generator, Iterable
from typing import Any, TYPE_CHECKING

from source_detection import SourceType, detect_source_type

if TYPE_CHECKING:
    from html_summarizer import WebArticleSummarizer
    from pdf_summarizer import PDFSummarizer
    from youtube_summarizer import YouTubeSummarizer


ProgressCallback = Callable[[str], None]


class SummarizerService:
    """Single application-facing entrypoint for all supported summary sources."""

    def __init__(
        self,
        pdf_summarizer: "PDFSummarizer | None" = None,
        web_summarizer: "WebArticleSummarizer | None" = None,
        youtube_summarizer: "YouTubeSummarizer | None" = None,
    ):
        if pdf_summarizer is None:
            from pdf_summarizer import PDFSummarizer

            pdf_summarizer = PDFSummarizer(max_chunk_length=2000)
        if web_summarizer is None:
            from html_summarizer import WebArticleSummarizer

            web_summarizer = WebArticleSummarizer(token_limit=3000, overlap_ratio=0.15)
        if youtube_summarizer is None:
            from youtube_summarizer import YouTubeSummarizer

            youtube_summarizer = YouTubeSummarizer()

        self.pdf_summarizer = pdf_summarizer
        self.web_summarizer = web_summarizer
        self.youtube_summarizer = youtube_summarizer

    @staticmethod
    def _yield_result(result: Any) -> Generator[str, None, None]:
        if result is None:
            return
        if isinstance(result, str):
            yield result
            return
        if isinstance(result, Iterable):
            for chunk in result:
                if chunk:
                    yield str(chunk)
            return
        yield str(result)

    def summarize_upload_stream(
        self,
        uploaded_file: Any,
        on_progress: ProgressCallback | None = None,
    ) -> Generator[str, None, None]:
        result = self.pdf_summarizer.get_summary(
            source=uploaded_file,
            on_progress=on_progress,
        )
        yield from self._yield_result(result)

    def summarize_stream(
        self,
        source: str,
        *,
        language: str = "auto",
        task_type: str = "long_summary",
    ) -> Generator[str, None, None]:
        source_type = detect_source_type(source)

        if source_type in {SourceType.YOUTUBE_URL, SourceType.VIDEO_FILE, SourceType.TRANSCRIPT_FILE}:
            yield from self.youtube_summarizer.get_summary_stream(source, language=language)
            return

        if source_type in {SourceType.HTML_URL, SourceType.PDF_URL}:
            yield from self.web_summarizer.get_summary(source, task_type=task_type)
            return

        if source_type == SourceType.PDF_FILE:
            result = self.pdf_summarizer.get_summary(source)
            yield from self._yield_result(result)
            return

        yield (
            f"無效的輸入: {source}\n"
            "請提供網頁 URL、PDF URL、PDF 檔案、本地影片、逐字稿檔案或 YouTube URL"
        )

    def summarize_sync(
        self,
        source: str,
        *,
        language: str = "auto",
        task_type: str = "long_summary",
    ) -> str:
        return "".join(
            self.summarize_stream(source, language=language, task_type=task_type)
        )
