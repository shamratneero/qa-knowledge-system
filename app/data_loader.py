from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.xlsx"


REQUIRED_COLUMNS = [
    "id",
    "question",
    "answer",
    "category",
    "keywords",
]


def load_knowledge_base():
    """Load and validate the knowledge base Excel file."""

    df = pd.read_excel(DATA_FILE)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Knowledge base is empty.")

    if df["question"].isnull().any():
        raise ValueError("Some questions are empty.")

    if df["answer"].isnull().any():
        raise ValueError("Some answers are empty.")

    return df
