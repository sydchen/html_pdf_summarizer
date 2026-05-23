import unittest

from source_detection import SourceType, detect_source_type, is_youtube_url


class SourceDetectionTests(unittest.TestCase):
    def test_detects_youtube_urls(self):
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=NgrCQcU0Sbg"))
        self.assertTrue(is_youtube_url("https://youtu.be/NgrCQcU0Sbg"))
        self.assertEqual(
            detect_source_type("https://www.youtube.com/watch?v=NgrCQcU0Sbg"),
            SourceType.YOUTUBE_URL,
        )

    def test_detects_web_and_pdf_urls(self):
        self.assertEqual(
            detect_source_type("https://example.com/paper.pdf"),
            SourceType.PDF_URL,
        )
        self.assertEqual(
            detect_source_type("https://example.com/article"),
            SourceType.HTML_URL,
        )

    def test_detects_local_files_by_suffix(self):
        self.assertEqual(detect_source_type("notes.srt"), SourceType.TRANSCRIPT_FILE)
        self.assertEqual(detect_source_type("lecture.txt"), SourceType.TRANSCRIPT_FILE)
        self.assertEqual(detect_source_type("talk.mp4"), SourceType.VIDEO_FILE)
        self.assertEqual(detect_source_type("document.pdf"), SourceType.PDF_FILE)

    def test_unknown_source(self):
        self.assertEqual(detect_source_type("archive.zip"), SourceType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
