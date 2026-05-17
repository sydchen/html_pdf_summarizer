import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from clean_transcript import clean_srt_file


class CleanTranscriptTests(unittest.TestCase):
    def clean_content(self, content: str) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.srt"
            input_path.write_text(content, encoding="utf-8")
            with redirect_stdout(StringIO()):
                return clean_srt_file(str(input_path), auto_output=False)

    def test_cleans_basic_srt_blocks(self):
        content = """1
00:00:00,000 --> 00:00:02,000
Hello world.

2
00:00:02,000 --> 00:00:04,000
This is another line.
"""

        self.assertEqual(
            self.clean_content(content),
            "Hello world.\nThis is another line.",
        )

    def test_preserves_numeric_subtitle_lines(self):
        content = """1
00:00:00,000 --> 00:00:02,000
The year was
2024
and everything changed.

2
00:00:02,000 --> 00:00:04,000
The next point follows.
"""

        self.assertEqual(
            self.clean_content(content),
            "The year was 2024 and everything changed.\nThe next point follows.",
        )

    def test_handles_utf8_bom_at_start(self):
        content = """\ufeff1
00:00:00,000 --> 00:00:02,000
First block should not be skipped.

2
00:00:02,000 --> 00:00:04,000
Second block.
"""

        self.assertEqual(
            self.clean_content(content),
            "First block should not be skipped.\nSecond block.",
        )

    def test_preserves_arrow_text_that_is_not_timestamp(self):
        content = """1
00:00:00,000 --> 00:00:02,000
The pipeline is input --> output.

2
00:00:02,000 --> 00:00:04,000
Done.
"""

        self.assertEqual(
            self.clean_content(content),
            "The pipeline is input --> output.\nDone.",
        )


if __name__ == "__main__":
    unittest.main()
