import re


def estimate_token_count(text: str, chars_per_token: int = 3) -> int:
    """Return a conservative token estimate without a tokenizer dependency."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return len(text) // chars_per_token


def split_text_by_estimated_tokens(text: str, max_tokens: int) -> list[str]:
    """Split text on paragraph and sentence boundaries using token estimates."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_tokens = estimate_token_count(paragraph)
        if paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+", paragraph)]
            for sentence in sentences:
                if not sentence:
                    continue
                sentence_tokens = estimate_token_count(sentence)
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                    current_tokens += sentence_tokens
            continue

        if current_tokens + paragraph_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_tokens = paragraph_tokens
        else:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
            current_tokens += paragraph_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def split_transcript_into_chunks(
    transcript: str,
    chunk_size: int = 8000,
    overlap_words: int = 200,
) -> list[str]:
    """Split transcript text on sentence boundaries with word overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")

    normalized = re.sub(r"\s+", " ", transcript).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence) + 1
        if current_sentences and current_length + sentence_length > chunk_size:
            current_text = " ".join(current_sentences).strip()
            chunks.append(current_text)

            overlap = " ".join(current_text.split()[-overlap_words:]) if overlap_words else ""
            current_sentences = [overlap, sentence] if overlap else [sentence]
            current_length = len(" ".join(current_sentences))
        else:
            current_sentences.append(sentence)
            current_length += sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks
