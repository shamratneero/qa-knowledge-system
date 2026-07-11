"""Edge-case tests for loader and API behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.loader import REQUIRED_COLUMNS, load_knowledge_base


def test_loader_rejects_empty_answers(tmp_path: Path):
    bad_file = tmp_path / "null_a.xlsx"
    pd.DataFrame(
        {
            "id": [1],
            "question": ["Q"],
            "answer": [None],
            "category": ["C"],
            "keywords": ["k"],
        }
    ).to_excel(bad_file, index=False)

    with pytest.raises(ValueError, match="answers are empty"):
        load_knowledge_base(bad_file)


def test_loader_rejects_no_rows(tmp_path: Path):
    empty_file = tmp_path / "no_rows.xlsx"
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_excel(empty_file, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_knowledge_base(empty_file)
