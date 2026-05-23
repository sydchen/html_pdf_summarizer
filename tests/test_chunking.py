import unittest

from chunking import (
    estimate_token_count,
    split_text_by_estimated_tokens,
    split_transcript_into_chunks,
)


class ChunkingTests(unittest.TestCase):
    def test_estimates_tokens_conservatively(self):
        self.assertEqual(estimate_token_count("abcdef"), 2)

    def test_splits_paragraphs_by_estimated_tokens(self):
        text = "aaa bbb ccc\n\n" + "ddd eee fff\n\n" + "ggg hhh iii"
        chunks = split_text_by_estimated_tokens(text, max_tokens=4)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.replace("\n\n", "") for chunk in chunks).replace(" ", ""), "aaabbbcccdddeeefffggghhhiii")

    def test_splits_transcript_with_overlap(self):
        transcript = "One sentence. Two sentence. Three sentence. Four sentence."
        chunks = split_transcript_into_chunks(transcript, chunk_size=28, overlap_words=2)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))
        self.assertIn("Two sentence", " ".join(chunks))

    def test_empty_transcript_returns_empty_list(self):
        self.assertEqual(split_transcript_into_chunks("   "), [])


if __name__ == "__main__":
    unittest.main()
