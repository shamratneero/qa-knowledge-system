"""Tests for deterministic and optional-LLM conversation summarization."""

from __future__ import annotations

import pandas as pd

from app.services import conversation_summary
from app.services.conversation_summary import (
    build_embedding_text,
    summarize_conversation,
    summarize_conversations,
)


def test_password_reset_example_matches_spec(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    conversation = (
        "Customer: Hi\n\n"
        "Agent: Hello\n\n"
        "Customer: I forgot my password and cannot login."
    )

    result = summarize_conversation(conversation, subject="Login issue")

    assert result["intent"] == "Password Reset"
    assert result["category"] == "Authentication"
    assert "password" in result["keywords"]
    assert "login" in result["keywords"] or "cannot" in result["keywords"]
    assert "password" in result["summary"].lower()
    assert result["summary"].startswith("Customer")


def test_greeting_only_conversations_collapse_to_same_representation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    conv_a = "Customer: Hi"
    conv_b = "Customer: Hello"
    conv_c = "Customer: Good morning"

    result_a = summarize_conversation(conv_a)
    result_b = summarize_conversation(conv_b)
    result_c = summarize_conversation(conv_c)

    assert result_a["summary"] == result_b["summary"] == result_c["summary"]
    assert result_a["intent"] == result_b["intent"] == result_c["intent"]
    assert result_a["category"] == result_b["category"] == result_c["category"]

    embedding_a = build_embedding_text(
        result_a["summary"], result_a["intent"], result_a["keywords"]
    )
    embedding_b = build_embedding_text(
        result_b["summary"], result_b["intent"], result_b["keywords"]
    )
    assert embedding_a == embedding_b


def test_summarize_conversations_adds_expected_columns(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    df = pd.DataFrame(
        {
            "ticket_id": ["1", "2"],
            "subject": ["Password reset", "Refund"],
            "conversation_text": [
                "Customer: Hi\n\nCustomer: I forgot my password and cannot login.",
                "Customer: I need a refund for my order urgently.",
            ],
        }
    )

    out = summarize_conversations(df)

    for col in [
        "summary",
        "intent",
        "category",
        "sentiment",
        "priority",
        "keywords",
        "entities",
        "product_names",
        "embedding_text",
    ]:
        assert col in out.columns

    assert out.loc[0, "intent"] == "Password Reset"
    assert out.loc[1, "intent"] == "Refund Request"
    assert out.loc[1, "priority"] == "high"


def test_summarize_conversations_empty_df_returns_copy():
    df = pd.DataFrame(columns=["ticket_id", "conversation_text"])
    out = summarize_conversations(df)
    assert out.empty


def test_urgent_negative_conversation_gets_high_priority(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    conversation = (
        "Customer: This is urgent, my account is broken and not working at all. "
        "I am furious and need this fixed immediately."
    )
    result = summarize_conversation(conversation)
    assert result["priority"] == "high"
    assert result["sentiment"] == "negative"


def test_llm_path_used_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SUMMARY_LLM_MODEL", "gpt-test")

    canned = {
        "summary": "Customer cannot access their account.",
        "intent": "Account Access",
        "category": "Authentication",
        "sentiment": "negative",
        "priority": "high",
        "keywords": ["account", "access"],
        "entities": [],
        "product_names": [],
    }

    monkeypatch.setattr(
        conversation_summary,
        "_generate_with_optional_llm",
        lambda text, subject: canned,
    )

    result = summarize_conversation("Customer: I cannot access my account.")
    assert result == canned


def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SUMMARY_LLM_MODEL", "gpt-test")

    class FakeOpenAIModule:
        class OpenAI:
            def __init__(self, api_key):
                raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        conversation_summary.importlib, "import_module", lambda name: FakeOpenAIModule
    )

    result = summarize_conversation(
        "Customer: I forgot my password and cannot login."
    )
    assert result["intent"] == "Password Reset"
