"""Unit tests for rag_core utilities (no Ollama or ChromaDB required)."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_core import split_into_chunks, build_prompt, make_metadata


class TestSplitIntoChunks:
    def test_empty_string_returns_empty(self):
        assert split_into_chunks("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_into_chunks("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = split_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_produces_multiple_chunks(self):
        # 1000 chars should produce more than 1 chunk with CHUNK_SIZE=500
        text = "a " * 500  # 1000 chars
        chunks = split_into_chunks(text)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        # With overlap, adjacent chunks should share content
        text = "word " * 300  # long enough to produce multiple chunks
        chunks = split_into_chunks(text)
        assert len(chunks) >= 2
        # End of first chunk should appear in beginning of second
        end_of_first = chunks[0][-50:]
        start_of_second = chunks[1][:50]
        # They should share some characters due to overlap
        assert any(c in start_of_second for c in end_of_first.split())

    def test_normalizes_whitespace(self):
        text = "hello    world\n\nfoo\tbar"
        chunks = split_into_chunks(text)
        assert "  " not in chunks[0]
        assert "\n" not in chunks[0]

    def test_no_empty_chunks(self):
        text = "a " * 400
        chunks = split_into_chunks(text)
        assert all(chunk.strip() for chunk in chunks)


class TestBuildPrompt:
    def test_contains_context(self):
        prompt = build_prompt("Some context here.", "What is this?")
        assert "Some context here." in prompt

    def test_contains_question(self):
        prompt = build_prompt("context", "What is the meaning?")
        assert "What is the meaning?" in prompt

    def test_context_before_question(self):
        prompt = build_prompt("MYCONTEXT", "MYQUESTION")
        assert prompt.index("MYCONTEXT") < prompt.index("MYQUESTION")

    def test_returns_string(self):
        assert isinstance(build_prompt("ctx", "q"), str)

    def test_empty_context(self):
        prompt = build_prompt("", "What is this?")
        assert "What is this?" in prompt


class TestMakeMetadata:
    def test_contains_source(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 0)
        assert meta["source"] == "file.pdf"

    def test_contains_uploaded_at(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 0)
        assert meta["uploaded_at"] == "2024-01-01T00:00:00"

    def test_contains_chunk_index(self):
        meta = make_metadata("file.pdf", "2024-01-01T00:00:00", 5)
        assert meta["chunk_index"] == 5

    def test_returns_dict(self):
        assert isinstance(make_metadata("f", "t", 0), dict)


class TestEmbedText:
    def test_calls_ollama_embeddings(self):
        with patch("rag_core.ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            from rag_core import embed_text
            result = embed_text("hello")
            mock_embed.assert_called_once()
            assert result == [0.1, 0.2, 0.3]


class TestGenerateAnswer:
    def test_calls_ollama_chat(self):
        with patch("rag_core.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "The answer."}}
            from rag_core import generate_answer
            result = generate_answer("some prompt")
            mock_chat.assert_called_once()
            assert result == "The answer."
