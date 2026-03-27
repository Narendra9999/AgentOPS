"""
Unit tests for document chunking strategies.
Run: python -m pytest tests/unit/test_chunking.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agentops_demo'))

from data_preparation.data_preprocessing.preprocessing.create_chunk import (
    clean_html, chunk_text)


# ── HTML cleaning ────────────────────────────────────────────

class TestCleanHTML:
    def test_strips_tags(self):
        html = "<div><p>Hello <b>world</b></p></div>"
        assert "Hello" in clean_html(html)
        assert "<div>" not in clean_html(html)

    def test_removes_scripts(self):
        html = "<div>Content<script>alert('xss')</script></div>"
        result = clean_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_removes_nav_footer(self):
        html = "<nav>Menu</nav><div>Main content</div><footer>Copyright</footer>"
        result = clean_html(html)
        assert "Menu" not in result
        assert "Copyright" not in result
        assert "Main content" in result

    def test_normalizes_whitespace(self):
        html = "<p>Line 1</p>\n\n\n\n\n<p>Line 2</p>"
        result = clean_html(html)
        assert "\n\n\n" not in result

    def test_empty_returns_empty(self):
        assert clean_html("") == ""
        assert clean_html(None) == ""


# ── Fixed strategy ───────────────────────────────────────────

class TestFixedChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Short text", chunk_size=100, overlap=20, strategy="fixed")
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_splits_at_exact_positions(self):
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=10, strategy="fixed")
        assert len(chunks) > 1
        assert all(len(c) <= 30 for c in chunks)

    def test_overlap_works(self):
        text = "ABCDEFGHIJ" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=30, overlap=10, strategy="fixed")
        # Second chunk should start 20 chars after first (30 - 10 overlap)
        assert chunks[1][:10] == chunks[0][20:30]


# ── Sentence strategy ────────────────────────────────────────

class TestSentenceChunking:
    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=40, overlap=0, strategy="sentence")
        # Should not cut mid-sentence
        for chunk in chunks:
            assert chunk.endswith(".") or chunk == chunks[-1]

    def test_short_text_single_chunk(self):
        text = "One sentence only."
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="sentence")
        assert len(chunks) == 1

    def test_empty_returns_empty(self):
        assert chunk_text("", strategy="sentence") == []


# ── Semantic strategy ────────────────────────────────────────

class TestSemanticChunking:
    def test_splits_at_paragraphs(self):
        text = "Paragraph one about Delta Lake.\n\nParagraph two about Unity Catalog.\n\nParagraph three about MLflow."
        chunks = chunk_text(text, chunk_size=200, overlap=0, strategy="semantic")
        # Should keep paragraphs together when they fit
        assert any("Delta Lake" in c for c in chunks)

    def test_long_paragraph_falls_back_to_sentence(self):
        long_para = "Sentence one. " * 50  # Very long paragraph
        text = f"Short para.\n\n{long_para}\n\nAnother short para."
        chunks = chunk_text(text, chunk_size=100, overlap=0, strategy="semantic")
        assert len(chunks) > 2  # Long paragraph should be sub-chunked

    def test_empty_returns_empty(self):
        assert chunk_text("", strategy="semantic") == []


# ── Strategy routing ─────────────────────────────────────────

class TestStrategyRouting:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("Some text", strategy="nonexistent")

    def test_all_strategies_produce_chunks(self):
        text = "First sentence. Second sentence. Third sentence." * 5
        for strategy in ["fixed", "sentence", "semantic"]:
            chunks = chunk_text(text, chunk_size=60, overlap=10, strategy=strategy)
            assert len(chunks) > 0, f"Strategy '{strategy}' produced no chunks"
