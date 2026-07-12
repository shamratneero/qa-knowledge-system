"""Conversation summarization: summary, intent, keywords, sentiment, priority.

Produces the semantic representation (summary + intent + keywords) that
downstream classification, clustering, and knowledge-base matching embed
instead of raw conversation text. This collapses low-information wording
variance (e.g. "Hi" vs "Hello" vs "Good morning") while keeping conversations
that share different wording but the same customer intent close together.

Works fully deterministically via rule-based NLP with no external dependency.
If OPENAI_API_KEY is configured, an LLM is used instead for richer summaries;
any LLM failure falls back to the deterministic path.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections import Counter
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.core.logging import logger

_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "sure",
    "yes",
    "no",
    "bye",
    "goodbye",
}

_AGENT_SENDER_HINTS = ("agent", "support", "bot", "staff", "team", "rep")

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Password Reset|Authentication", ("password", "login", "log in", "locked out", "authentication", "reset my")),
    ("Refund Request|Billing", ("refund", "money back", "chargeback", "reimburse")),
    ("Cancellation Request|Billing", ("cancel", "cancellation", "cancel my")),
    ("Booking Inquiry|Reservations", ("booking", "reservation", "book a", "reschedule")),
    ("Parking Inquiry|Facilities", ("parking", "park my car", "garage")),
    ("Billing Inquiry|Billing", ("invoice", "bill", "charge", "payment", "overcharged")),
    ("Technical Issue|Technical", ("error", "bug", "not working", "broken", "crash", "doesn't work", "won't load")),
]

_POSITIVE_WORDS = {
    "thanks",
    "thank",
    "great",
    "good",
    "awesome",
    "excellent",
    "happy",
    "appreciate",
    "perfect",
    "resolved",
}
_NEGATIVE_WORDS = {
    "angry",
    "frustrated",
    "terrible",
    "bad",
    "worst",
    "hate",
    "broken",
    "not working",
    "disappointed",
    "furious",
    "unacceptable",
    "cannot",
    "can't",
    "never",
    "annoyed",
}
_URGENT_WORDS = {
    "urgent",
    "asap",
    "immediately",
    "right now",
    "cannot access",
    "can't access",
    "not working",
    "broken",
    "angry",
    "furious",
    "emergency",
}

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+){0,2})\b")
_CODE_RE = re.compile(r"\B#\w+|\b[A-Z]{1,4}-?\d{3,}\b")
_LINE_RE = re.compile(r"^(?P<speaker>[^:]{1,60}):\s*(?P<message>.*)$")


def summarize_conversation(conversation_text: str, subject: str = "") -> dict[str, Any]:
    """Summarize a single reconstructed conversation into structured intelligence."""
    text = str(conversation_text or "")
    subj = str(subject or "")

    llm_result = _generate_with_optional_llm(text, subj)
    if llm_result is not None:
        return llm_result

    return _summarize_deterministic(text, subj)


def summarize_conversations(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Add summary/intent/keywords/etc. and embedding_text columns to a conversations df."""
    if conversations_df.empty:
        return conversations_df.copy()

    df = conversations_df.copy()
    summaries: list[str] = []
    intents: list[str] = []
    categories: list[str] = []
    sentiments: list[str] = []
    priorities: list[str] = []
    keywords_col: list[str] = []
    entities_col: list[str] = []
    product_names_col: list[str] = []
    embedding_texts: list[str] = []

    for _, row in df.iterrows():
        result = summarize_conversation(
            conversation_text=str(row.get("conversation_text", "")),
            subject=str(row.get("subject", "")),
        )
        summaries.append(result["summary"])
        intents.append(result["intent"])
        categories.append(result["category"])
        sentiments.append(result["sentiment"])
        priorities.append(result["priority"])
        keywords_col.append(", ".join(result["keywords"]))
        entities_col.append(", ".join(result["entities"]))
        product_names_col.append(", ".join(result["product_names"]))
        embedding_texts.append(
            build_embedding_text(result["summary"], result["intent"], result["keywords"])
        )

    df["summary"] = summaries
    df["intent"] = intents
    df["category"] = categories
    df["sentiment"] = sentiments
    df["priority"] = priorities
    df["keywords"] = keywords_col
    df["entities"] = entities_col
    df["product_names"] = product_names_col
    df["embedding_text"] = embedding_texts

    logger.info("conversation_summary completed rows=%d", len(df))
    return df


def build_embedding_text(summary: str, intent: str, keywords: list[str]) -> str:
    """Build the semantic representation embedded downstream in place of raw text."""
    keyword_str = ", ".join(keywords)
    return f"{summary} Intent: {intent}. Keywords: {keyword_str}.".strip()


# ---------------------------------------------------------------------------
# Deterministic path
# ---------------------------------------------------------------------------


def _summarize_deterministic(text: str, subject: str) -> dict[str, Any]:
    customer_lines = _extract_customer_lines(text)
    informative_lines = [line for line in customer_lines if not _is_greeting(line)]

    summary = _build_summary(informative_lines, subject)
    # Only informative content (never raw greeting text) feeds keyword/intent/
    # sentiment analysis, so purely trivial conversations ("Hi" / "Hello" /
    # "Good morning") always collapse to the same semantic representation.
    analysis_text = " ".join(informative_lines) or subject
    keywords = _extract_keywords(analysis_text)
    intent, category = _classify_intent_and_category(analysis_text, keywords)
    sentiment = _sentiment(analysis_text)
    priority = _priority(sentiment, category, analysis_text)
    entities, product_names = _extract_entities(text)

    return {
        "summary": summary,
        "intent": intent,
        "category": category,
        "sentiment": sentiment,
        "priority": priority,
        "keywords": keywords,
        "entities": entities,
        "product_names": product_names,
    }


def _extract_customer_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        match = _LINE_RE.match(raw_line.strip())
        if not match:
            continue
        speaker = match.group("speaker").strip().lower()
        message = match.group("message").strip()
        if not message:
            continue
        if any(hint in speaker for hint in _AGENT_SENDER_HINTS):
            continue
        lines.append(message)
    return lines


def _is_greeting(line: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", line.lower()).strip()
    if not normalized:
        return True
    if normalized in _GREETING_PHRASES:
        return True
    words = normalized.split()
    return len(words) <= 2 and all(w in _GREETING_PHRASES for w in words)


def _build_summary(informative_lines: list[str], subject: str) -> str:
    if not informative_lines:
        return "Customer opened a conversation with a general greeting."

    ranked = sorted(informative_lines, key=len, reverse=True)[:3]
    ranked_in_order = [line for line in informative_lines if line in ranked][:3]
    body = " ".join(ranked_in_order).strip()
    if not body:
        return "Customer opened a conversation with a general greeting."

    if not body.endswith((".", "!", "?")):
        body += "."
    return f"Customer reported: {body}"


def _extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", text.lower())
    filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:max_keywords]]


def _classify_intent_and_category(text: str, keywords: list[str]) -> tuple[str, str]:
    haystack = text.lower()
    for label, triggers in _INTENT_RULES:
        if any(trigger in haystack for trigger in triggers):
            intent, category = label.split("|")
            return intent, category
    return "General Inquiry", "General"


def _sentiment(text: str) -> str:
    haystack = text.lower()
    positive = sum(1 for w in _POSITIVE_WORDS if w in haystack)
    negative = sum(1 for w in _NEGATIVE_WORDS if w in haystack)
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "neutral"


def _priority(sentiment: str, category: str, text: str) -> str:
    haystack = text.lower()
    is_urgent = any(word in haystack for word in _URGENT_WORDS)
    if is_urgent or sentiment == "negative":
        return "high"
    if category in {"Billing", "Technical", "Reservations"}:
        return "medium"
    return "low"


def _extract_entities(text: str) -> tuple[list[str], list[str]]:
    codes = sorted(set(_CODE_RE.findall(text)))
    proper_nouns = [
        m.strip()
        for m in _PROPER_NOUN_RE.findall(text)
        if m.strip().lower() not in {"customer", "agent", "guest", "support agent"}
    ]
    seen: set[str] = set()
    unique_nouns: list[str] = []
    for noun in proper_nouns:
        key = noun.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_nouns.append(noun)

    entities = (unique_nouns[:5] + codes)[:8]
    product_names = unique_nouns[:3]
    return entities, product_names


# ---------------------------------------------------------------------------
# Optional LLM path
# ---------------------------------------------------------------------------

_REQUIRED_LLM_FIELDS = (
    "summary",
    "intent",
    "category",
    "sentiment",
    "priority",
    "keywords",
    "entities",
    "product_names",
)


def _generate_with_optional_llm(text: str, subject: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("SUMMARY_LLM_MODEL") or os.getenv("AI_ASSISTANT_LLM_MODEL")
    if not api_key or not model:
        return None

    try:
        openai_module = importlib.import_module("openai")
        OpenAI = getattr(openai_module, "OpenAI")
    except Exception:
        logger.warning(
            "LLM summarization requested but openai package is unavailable; using deterministic mode."
        )
        return None

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Summarize this customer support conversation. Return STRICT JSON only, "
            "with keys: summary (1-3 sentences), intent (short label), category "
            "(short label), sentiment (positive|neutral|negative), priority "
            "(low|medium|high), keywords (array of strings), entities (array of "
            "strings), product_names (array of strings).\n"
            f"Subject: {subject}\n"
            f"Conversation:\n{text}"
        )
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured intelligence from support conversations. Reply with strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = _parse_llm_json(content)
        if parsed is None:
            return None
        return parsed
    except Exception:
        logger.exception("Optional LLM summarization failed; falling back to deterministic summary.")
        return None


def _parse_llm_json(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(data, dict) or not all(k in data for k in _REQUIRED_LLM_FIELDS):
        return None

    try:
        return {
            "summary": str(data["summary"]),
            "intent": str(data["intent"]),
            "category": str(data["category"]),
            "sentiment": str(data["sentiment"]),
            "priority": str(data["priority"]),
            "keywords": [str(k) for k in data["keywords"]],
            "entities": [str(k) for k in data["entities"]],
            "product_names": [str(k) for k in data["product_names"]],
        }
    except Exception:
        return None


__all__ = [
    "build_embedding_text",
    "summarize_conversation",
    "summarize_conversations",
]
