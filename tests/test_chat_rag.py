"""Unit tests for chat_rag input sanitization."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chat_rag import sanitize_query


class TestSanitizeQuery:
    def test_valid_query(self):
        query, error = sanitize_query("What is machine learning?")
        assert query == "What is machine learning?"
        assert error is None

    def test_strips_whitespace(self):
        query, error = sanitize_query("  hello  ")
        assert query == "hello"
        assert error is None

    def test_empty_string_returns_error(self):
        query, error = sanitize_query("")
        assert query is None
        assert error is not None

    def test_whitespace_only_returns_error(self):
        query, error = sanitize_query("   ")
        assert query is None
        assert error is not None

    def test_query_at_max_length_is_valid(self):
        from config import MAX_QUERY_LENGTH
        query, error = sanitize_query("a" * MAX_QUERY_LENGTH)
        assert error is None

    def test_query_over_max_length_returns_error(self):
        from config import MAX_QUERY_LENGTH
        query, error = sanitize_query("a" * (MAX_QUERY_LENGTH + 1))
        assert query is None
        assert error is not None
