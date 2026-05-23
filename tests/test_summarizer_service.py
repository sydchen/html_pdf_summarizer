import unittest

from summarizer_service import SummarizerService


class FakePdfSummarizer:
    def get_summary(self, source, on_progress=None):
        if on_progress:
            on_progress("pdf progress")
        return f"pdf:{source}"


class FakeWebSummarizer:
    def get_summary(self, source, task_type="long_summary"):
        yield f"web:{task_type}:{source}"


class FakeYouTubeSummarizer:
    def get_summary_stream(self, source, language="auto"):
        yield f"youtube:{language}:{source}"


class SummarizerServiceTests(unittest.TestCase):
    def make_service(self):
        return SummarizerService(
            pdf_summarizer=FakePdfSummarizer(),
            web_summarizer=FakeWebSummarizer(),
            youtube_summarizer=FakeYouTubeSummarizer(),
        )

    def test_routes_web_url_to_web_summarizer(self):
        service = self.make_service()

        result = service.summarize_sync("https://example.com/article", task_type="short_summary")

        self.assertEqual(result, "web:short_summary:https://example.com/article")

    def test_routes_youtube_url_to_youtube_summarizer(self):
        service = self.make_service()

        result = service.summarize_sync("https://youtu.be/NgrCQcU0Sbg", language="en")

        self.assertEqual(result, "youtube:en:https://youtu.be/NgrCQcU0Sbg")

    def test_routes_pdf_file_to_pdf_summarizer(self):
        service = self.make_service()

        result = service.summarize_sync("paper.pdf")

        self.assertEqual(result, "pdf:paper.pdf")

    def test_upload_summary_reports_progress(self):
        service = self.make_service()
        progress = []

        result = "".join(service.summarize_upload_stream("upload", on_progress=progress.append))

        self.assertEqual(result, "pdf:upload")
        self.assertEqual(progress, ["pdf progress"])


if __name__ == "__main__":
    unittest.main()
