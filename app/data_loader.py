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


def load_knowledge_base(file_path=None):
    """Load and validate the knowledge base Excel file.

    Args:
        file_path (str | Path | None): Optional path to the Excel file. If
            not provided, the default `DATA_FILE` is used.
    """

    data_path = Path(file_path) if file_path is not None else DATA_FILE

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_excel(data_path)

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
