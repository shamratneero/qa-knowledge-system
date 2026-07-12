"""Deterministic cluster labeling for semantic conversation clusters.

Labels are built in strict priority order so they describe CUSTOMER
PROBLEMS, never people's names, greetings, or random nouns:

    1. Intent    (from app.services.conversation_summary -- fixed, rule-derived
                   vocabulary, e.g. "Password Reset")
    2. Category  (same source, e.g. "Authentication")
    3. Summary   (per-conversation summary text, sanitized and reduced to
                   its top terms)
    4. Keywords  (precomputed per-conversation keyword list, sanitized)
    5. Sanitized raw-text TF-IDF -- legacy fallback used only when a cluster
       carries none of the columns above (e.g. callers that build a cluster
       DataFrame directly from raw subject/conversation_text without running
       the summarization stage first).

Tiers 1-4 consume text that has already been produced by
conversation_summary.py, which never includes raw speaker names or PII.
Tier 5 additionally strips speaker-label prefixes, PII, and probable
person/company names before generating a label, since it is the one tier
that still touches raw conversation text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.services.conversation_summary import (
    LABEL_SAFE_TERMS,
    is_probable_person_name,
    strip_pii,
)

_GENERIC_SUMMARY_SENTINEL = "customer opened a conversation with a general greeting"
_SPEAKER_LINE_RE = re.compile(r"^[^:\n]{1,60}:\s*")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']{2,}")
_LABEL_SAFE_TOKENS = {
    token for phrase in LABEL_SAFE_TERMS for token in phrase.split()
}
_RAW_TEXT_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _LABEL_SAFE_TOKENS)


class ClusterLabelGenerator(Protocol):
    """Interface for pluggable cluster label generators (e.g., future LLM-backed generators)."""

    def generate_label(self, cluster_id: int, cluster_df: pd.DataFrame) -> str:
        """Generate a concise human-readable label for one cluster."""


@dataclass(slots=True)
class DeterministicClusterLabelGenerator:
    """Generate cluster labels using intent/category/summary/keywords, in that
    priority order, falling back to sanitized keyword extraction over raw
    text only when none of those columns are available."""

    max_terms: int = 2
    max_label_chars: int = 100

    def generate_label(self, cluster_id: int, cluster_df: pd.DataFrame) -> str:
        if cluster_df.empty:
            return f"Cluster {cluster_id}"

        for tier in (
            self._label_from_column(cluster_df, "intent"),
            self._label_from_column(cluster_df, "category"),
            self._label_from_summary(cluster_df),
            self._label_from_keywords_column(cluster_df),
            self._label_from_raw_text(cluster_df),
        ):
            if tier:
                return tier[: self.max_label_chars]

        return f"Cluster {cluster_id}"

    # -- Tier 1 & 2: intent / category -----------------------------------

    def _label_from_column(self, cluster_df: pd.DataFrame, column: str) -> str | None:
        if column not in cluster_df.columns:
            return None
        values = [
            str(v).strip()
            for v in cluster_df[column].tolist()
            if str(v or "").strip()
        ]
        if not values:
            return None
        mode = Counter(values).most_common(1)[0][0]
        return mode or None

    # -- Tier 3: summary ----------------------------------------------------

    def _label_from_summary(self, cluster_df: pd.DataFrame) -> str | None:
        if "summary" not in cluster_df.columns:
            return None
        summaries = [
            str(v).strip()
            for v in cluster_df["summary"].tolist()
            if str(v or "").strip()
            and str(v).strip().lower().rstrip(".") != _GENERIC_SUMMARY_SENTINEL
        ]
        if not summaries:
            return None

        representative = Counter(summaries).most_common(1)[0][0]
        body = re.sub(r"^customer reported:\s*", "", representative, flags=re.IGNORECASE)
        body = strip_pii(body).rstrip(".")

        terms = self._top_terms(body, self.max_terms)
        if not terms:
            return None
        return " / ".join(self._title_case_term(t) for t in terms)

    # -- Tier 4: keywords column ---------------------------------------------

    def _label_from_keywords_column(self, cluster_df: pd.DataFrame) -> str | None:
        if "keywords" not in cluster_df.columns:
            return None
        counts: Counter[str] = Counter()
        for raw in cluster_df["keywords"].tolist():
            for term in str(raw or "").split(","):
                normalized = term.strip().lower()
                if not normalized or normalized in LABEL_SAFE_TERMS:
                    continue
                if is_probable_person_name(term.strip(), sentence_initial=False):
                    continue
                counts[normalized] += 1

        if not counts:
            return None
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_terms = [term for term, _ in ranked[: self.max_terms]]
        return " / ".join(self._title_case_term(t) for t in top_terms)

    # -- Tier 5: sanitized raw-text TF-IDF (legacy fallback) -----------------

    def _label_from_raw_text(self, cluster_df: pd.DataFrame) -> str | None:
        documents = self._build_sanitized_documents(cluster_df)
        if not documents:
            return None

        terms = self._extract_top_terms(documents)
        if not terms:
            return None
        return " / ".join(self._title_case_term(t) for t in terms[: self.max_terms])

    def _build_sanitized_documents(self, cluster_df: pd.DataFrame) -> list[str]:
        documents: list[str] = []
        for _, row in cluster_df.iterrows():
            parts = [
                str(row.get("subject", "") or "").strip(),
                str(row.get("conversation_text", "") or "").strip(),
            ]
            text = "\n".join(p for p in parts if p)
            if not text:
                continue

            # Strip "Speaker: " line prefixes -- the primary source of leaked
            # names in raw conversation text -- then blank out PII.
            stripped_lines = [_SPEAKER_LINE_RE.sub("", line) for line in text.splitlines()]
            sanitized = strip_pii("\n".join(stripped_lines))
            if sanitized:
                documents.append(sanitized)
        return documents

    def _extract_top_terms(self, texts: list[str]) -> list[str]:
        # TfidfVectorizer lowercases everything, which would destroy the
        # capitalization the name heuristic depends on -- so the banned-name
        # token set must be computed from the ORIGINAL (still-cased) text,
        # before vectorization, then applied to the (lowercased) ranked terms.
        banned_name_tokens: set[str] = set()
        for text in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                words = sentence.split()
                for idx, word in enumerate(words):
                    if is_probable_person_name(word, sentence_initial=(idx == 0)):
                        banned_name_tokens.add(word.strip(".,!?").lower())

        vectorizer = TfidfVectorizer(
            stop_words=_RAW_TEXT_STOP_WORDS,
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
            max_features=1500,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Raised by scikit-learn when every token across all documents is
            # filtered out as a stopword/name (e.g. "Thanks so much, regards")
            # -- there is simply no usable label material at this tier.
            return []
        if matrix.shape[1] == 0:
            return []

        mean_scores = matrix.mean(axis=0).A1
        features = vectorizer.get_feature_names_out()

        ranked = sorted(
            ((features[idx], float(mean_scores[idx])) for idx in range(len(features))),
            key=lambda item: (-item[1], item[0]),
        )

        unique_terms: list[str] = []
        seen_tokens: set[str] = set()
        for term, _ in ranked:
            normalized = term.strip().lower()
            if not normalized:
                continue
            # Never emit a term containing a probable person/company name or
            # a generic/greeting word -- unknown proper nouns are excluded by
            # default at this fallback tier; only the controlled intent/
            # category vocabulary (tiers 1-2) is trusted to surface them.
            words = normalized.split()
            if any(w in LABEL_SAFE_TERMS or w in banned_name_tokens for w in words):
                continue
            token_key = tuple(sorted(words))
            token_hash = " ".join(token_key)
            if token_hash in seen_tokens:
                continue
            seen_tokens.add(token_hash)
            unique_terms.append(normalized)
            if len(unique_terms) >= max(self.max_terms * 4, 6):
                break

        return unique_terms

    # -- shared helpers -------------------------------------------------

    def _top_terms(self, text: str, max_terms: int) -> list[str]:
        banned = set()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            words = sentence.split()
            for idx, word in enumerate(words):
                if is_probable_person_name(word, sentence_initial=(idx == 0)):
                    banned.add(word.strip(".,!?").lower())

        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        filtered = [
            t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in LABEL_SAFE_TERMS and t not in banned
        ]
        if not filtered:
            return []
        counts = Counter(filtered)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [term for term, _ in ranked[:max_terms]]

    def _title_case_term(self, term: str) -> str:
        cleaned = re.sub(r"\s+", " ", term).strip()
        return " ".join(part.capitalize() for part in cleaned.split(" "))


def assign_cluster_labels(
    clustered_df: pd.DataFrame,
    generator: ClusterLabelGenerator | None = None,
) -> pd.DataFrame:
    """Assign deterministic cluster labels while preserving numeric cluster IDs."""
    if clustered_df.empty:
        return clustered_df.copy()

    if "cluster_id" not in clustered_df.columns:
        return clustered_df.copy()

    df = clustered_df.copy()
    label_generator = generator or DeterministicClusterLabelGenerator()

    labels_by_id: dict[int, str] = {}
    for cluster_id, group in df.groupby("cluster_id", sort=True):
        cid = int(cluster_id)
        generated = label_generator.generate_label(cid, group)
        labels_by_id[cid] = generated or f"Cluster {cid}"

    df["cluster_label"] = [
        labels_by_id.get(int(cid), f"Cluster {int(cid)}")
        for cid in df["cluster_id"].tolist()
    ]
    return df


__all__ = [
    "ClusterLabelGenerator",
    "DeterministicClusterLabelGenerator",
    "assign_cluster_labels",
]
