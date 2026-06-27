import unittest

from transcriber import get_apple_podcast_filename, get_apple_podcast_id


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


if __name__ == "__main__":
    unittest.main()
