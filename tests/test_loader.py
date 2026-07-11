"""Tests for the knowledge base data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.loader import REQUIRED_COLUMNS, load_knowledge_base


def test_load_default_knowledge_base():
    df = load_knowledge_base()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path / "missing.xlsx")


def test_load_missing_columns(tmp_path: Path):
    bad_file = tmp_path / "bad.xlsx"
    pd.DataFrame({"id": [1], "question": ["Q"]}).to_excel(bad_file, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_knowledge_base(bad_file)


def test_load_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.xlsx"
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_excel(empty_file, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_knowledge_base(empty_file)


def test_load_null_questions(tmp_path: Path):
    bad_file = tmp_path / "null_q.xlsx"
    pd.DataFrame(
        {
            "id": [1],
            "question": [None],
            "answer": ["A"],
            "category": ["C"],
            "keywords": ["k"],
        }
    ).to_excel(bad_file, index=False)
    with pytest.raises(ValueError, match="questions are empty"):
        load_knowledge_base(bad_file)
