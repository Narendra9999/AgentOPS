"""
Data Preprocessing — HTML cleaning and document chunking.
Supports multiple chunking strategies: fixed, sentence, semantic.
"""

from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def clean_html(html_text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────
# Strategy: fixed — split at exact character boundaries
# ──────────────────────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text at fixed character positions with overlap."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────
# Strategy: sentence — split at sentence boundaries
# ──────────────────────────────────────────────────────────────────

def chunk_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split at sentence boundaries (.!?) respecting chunk_size."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ──────────────────────────────────────────────────────────────────
# Strategy: semantic — split at paragraph/section boundaries
# ──────────────────────────────────────────────────────────────────

def chunk_semantic(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split at semantic boundaries (double newlines = paragraphs/sections).
    Falls back to sentence splitting if paragraphs exceed chunk_size.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    # Split on paragraph boundaries (double newline)
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds chunk_size, sub-chunk it by sentence
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            sub_chunks = chunk_sentence(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += ("\n\n" + para if current_chunk else para)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ──────────────────────────────────────────────────────────────────
# Main entry point — routes to the selected strategy
# ──────────────────────────────────────────────────────────────────

STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "sentence",
) -> list[str]:
    """
    Split text into overlapping chunks using the specified strategy.

    Args:
        text: Input text
        chunk_size: Max characters per chunk
        overlap: Overlap characters between chunks
        strategy: "fixed", "sentence", or "semantic"

    Returns:
        List of text chunks
    """
    chunk_fn = STRATEGIES.get(strategy)
    if not chunk_fn:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from: {list(STRATEGIES.keys())}")

    return chunk_fn(text, chunk_size, overlap)


def process_document(url: str, html_text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                     overlap: int = DEFAULT_CHUNK_OVERLAP, strategy: str = "sentence") -> list[dict]:
    """Clean HTML and chunk into records ready for vector search."""
    clean_text = clean_html(html_text)
    if not clean_text:
        return []

    chunks = chunk_text(clean_text, chunk_size, overlap, strategy)

    return [
        {
            "url": url,
            "chunk_id": f"{url}_{i}",
            "chunk_index": i,
            "chunk_text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]
