import unittest
import tempfile
from pathlib import Path

from transcriber import (
    cleanup_source_audio,
    get_apple_podcast_filename,
    get_apple_podcast_id,
    normalize_transcript_filename,
    standard_transcript_path,
)


class TranscriberTests(unittest.TestCase):
    def test_apple_podcast_filename_uses_podcast_slug(self):
        url = "https://podcasts.apple.com/us/podcast/socks-and-sandals/id1601670119?i=1000772202936"

        self.assertEqual(get_apple_podcast_filename(url), "socks-and-sandals")

    def test_apple_podcast_id_alias_keeps_existing_imports_working(self):
        url = "https://podcasts.apple.com/us/podcast/socks-and-sandals/id1601670119?i=1000772202936"

        self.assertEqual(get_apple_podcast_id(url), "socks-and-sandals")

    def test_apple_podcast_filename_falls_back_to_episode_id(self):
        url = "https://podcasts.apple.com/us/id1601670119?i=1000772202936"

        self.assertEqual(get_apple_podcast_filename(url), "1000772202936")

    def test_standard_transcript_path_omits_wav_suffix(self):
        self.assertEqual(
            standard_transcript_path("youtube_downloads", "old-spice", "en"),
            Path("youtube_downloads/old-spice.en.srt"),
        )

    def test_standard_transcript_path_uses_auto_for_auto_language(self):
        self.assertEqual(
            standard_transcript_path("youtube_downloads", "old-spice", "auto"),
            Path("youtube_downloads/old-spice.auto.srt"),
        )

    def test_normalize_transcript_filename_removes_wav_component(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "old-spice.wav.en.srt"
            legacy_path.write_text("subtitle", encoding="utf-8")

            result = normalize_transcript_filename(
                str(legacy_path),
                tmp_dir,
                "old-spice",
                "en",
            )

            self.assertEqual(Path(result), Path(tmp_dir) / "old-spice.en.srt")
            self.assertFalse(legacy_path.exists())
            self.assertEqual(Path(result).read_text(encoding="utf-8"), "subtitle")

    def test_cleanup_source_audio_removes_mp3_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mp3_path = Path(tmp_dir) / "old-spice.mp3"
            mp3_path.write_text("audio", encoding="utf-8")

            cleanup_source_audio(str(mp3_path))

            self.assertFalse(mp3_path.exists())

    def test_cleanup_source_audio_keeps_mp3_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mp3_path = Path(tmp_dir) / "old-spice.mp3"
            mp3_path.write_text("audio", encoding="utf-8")

            cleanup_source_audio(str(mp3_path), keep_source_audio=True)

            self.assertTrue(mp3_path.exists())

    def test_cleanup_source_audio_does_not_remove_non_mp3(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            m4a_path = Path(tmp_dir) / "old-spice.m4a"
            m4a_path.write_text("audio", encoding="utf-8")

            cleanup_source_audio(str(m4a_path))

            self.assertTrue(m4a_path.exists())


if __name__ == "__main__":
    unittest.main()
