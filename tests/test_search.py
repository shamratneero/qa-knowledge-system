"""Tests for search engines."""

from __future__ import annotations

import pytest

from app.search.engine import search
from app.search.fuzzy import search_fuzzy
from app.search.hybrid import search_hybrid
from app.search.semantic import search_semantic


class TestKeywordSearch:
    def test_finds_exact_match(self, sample_questions):
        result = search(sample_questions["ai"])
        assert result["found"] is True
        assert result["method"] == "keyword"
        assert result["results"][0]["question"] == "What is AI?"

    def test_returns_confidence(self, sample_questions):
        result = search(sample_questions["ai"])
        assert 0 < result["results"][0]["confidence"] <= 1

    def test_no_match(self, sample_questions):
        result = search(sample_questions["missing"])
        assert result["found"] is False

    def test_empty_query(self, sample_questions):
        result = search(sample_questions["empty"])
        assert result["found"] is False
        assert "Empty" in result["message"]


class TestFuzzySearch:
    def test_handles_typos(self, sample_questions):
        result = search_fuzzy(sample_questions["typo"])
        assert result["found"] is True
        assert result["method"] == "fuzzy"

    def test_no_match(self, sample_questions):
        result = search_fuzzy(sample_questions["missing"])
        assert result["found"] is False


@pytest.mark.slow
class TestSemanticSearch:
    def test_finds_paraphrase(self, sample_questions):
        result = search_semantic(sample_questions["paraphrase"])
        assert result["found"] is True
        assert result["method"] == "semantic"
        assert result["results"][0]["confidence"] > 0

    def test_no_match_on_gibberish(self, sample_questions):
        result = search_semantic(sample_questions["missing"], threshold=0.9)
        assert result["found"] is False


@pytest.mark.slow
class TestHybridSearch:
    def test_default_hybrid(self, sample_questions):
        result = search_hybrid(sample_questions["ml"])
        assert result["found"] is True
        assert result["method"] == "hybrid"
        assert "weights" in result

    def test_keyword_method_routes(self, sample_questions):
        result = search_hybrid(sample_questions["ai"], method="keyword")
        assert result["method"] == "keyword"

    def test_unknown_method(self, sample_questions):
        result = search_hybrid(sample_questions["ai"], method="unknown")  # type: ignore[arg-type]
        assert result["found"] is False
