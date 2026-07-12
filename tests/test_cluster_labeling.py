"""Tests for deterministic cluster label generation."""

from __future__ import annotations

import pandas as pd

from app.services.cluster_labeling import (
    DeterministicClusterLabelGenerator,
    assign_cluster_labels,
)


def test_assign_cluster_labels_generates_intent_terms():
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2", "T3", "T4"],
            "subject": [
                "Password reset needed",
                "Cannot reset password",
                "Refund request",
                "Billing refund not received",
            ],
            "conversation_text": [
                "Guest forgot password and needs account reset link.",
                "Please help reset password for my account.",
                "I want a refund for duplicate charge.",
                "Need billing refund because charge was wrong.",
            ],
            "cluster_id": [0, 0, 1, 1],
        }
    )

    labeled = assign_cluster_labels(df)

    labels = labeled.groupby("cluster_id")["cluster_label"].first().to_dict()
    assert 0 in labels and 1 in labels
    assert "password" in labels[0].lower()
    assert "refund" in labels[1].lower()


def test_assign_cluster_labels_fallback_when_empty_text():
    df = pd.DataFrame(
        {
            "ticket_id": ["T1"],
            "subject": [""],
            "conversation_text": [""],
            "cluster_id": [3],
        }
    )

    labeled = assign_cluster_labels(df)
    assert labeled.iloc[0]["cluster_label"] == "Cluster 3"


def test_deterministic_label_generator_is_stable():
    generator = DeterministicClusterLabelGenerator()
    cluster_df = pd.DataFrame(
        {
            "subject": ["Login OTP failure", "OTP not received"],
            "conversation_text": [
                "OTP code is not received during login.",
                "Login fails because otp is missing.",
            ],
        }
    )

    label1 = generator.generate_label(2, cluster_df)
    label2 = generator.generate_label(2, cluster_df)

    assert label1 == label2
    assert label1


def test_label_prefers_intent_over_names_in_raw_text():
    """Reproduces the reported bug: a cluster whose raw conversation text is
    full of speaker names/sign-offs must label from the already-computed
    intent column, never from a name (e.g. "Roger / Guest")."""
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2", "T3"],
            "subject": ["Login issue", "Cannot login", "Password trouble"],
            "conversation_text": [
                "Roger: Hi\n\nSupport Agent: Hello\n\nRoger: I forgot my password. Thanks, Jackie",
                "Matthew: This is Matthew. I forgot my password too, cannot access account.",
                "Jackie: I cannot login, my password is not working. Regards, Roger",
            ],
            "intent": ["Password Reset", "Password Reset", "Password Reset"],
            "category": ["Authentication", "Authentication", "Authentication"],
            "cluster_id": [5, 5, 5],
        }
    )

    labeled = assign_cluster_labels(df)
    label = labeled.iloc[0]["cluster_label"]

    assert label == "Password Reset"
    for banned in ("roger", "jackie", "matthew", "guest", "thank"):
        assert banned not in label.lower()


def test_label_falls_back_to_category_when_intent_missing():
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2"],
            "subject": ["Billing", "Billing"],
            "conversation_text": [
                "Roger: I was overcharged.",
                "Jackie: Billing issue.",
            ],
            "intent": ["", ""],
            "category": ["Billing", "Billing"],
            "cluster_id": [7, 7],
        }
    )

    labeled = assign_cluster_labels(df)
    assert labeled.iloc[0]["cluster_label"] == "Billing"


def test_raw_text_fallback_never_emits_names_or_greetings():
    """Adversarial fixture for the legacy tier-5 fallback (no intent/category/
    summary/keywords columns available at all)."""
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2", "T3"],
            "subject": ["Refund needed", "Refund please", "Refund request"],
            "conversation_text": [
                "Roger: Hi\n\nAgent: Hello\n\nRoger: I need a refund for my order. Thanks, Jackie",
                "Matthew: This is Matthew, please refund my duplicate charge. Regards, Roger",
                "Jackie: Refund my payment, it was charged twice. Best, Matthew",
            ],
            "cluster_id": [9, 9, 9],
        }
    )

    labeled = assign_cluster_labels(df)
    label = labeled.iloc[0]["cluster_label"]

    # "refund" appears in every document so TF-IDF naturally deprioritizes it
    # in favor of more distinctive terms (charge/duplicate/payment) -- the
    # real assertion here is that no name/greeting ever leaks into the label.
    assert label and label != "Cluster 9"
    for banned in ("roger", "jackie", "matthew", "thank", "hi", "hello"):
        assert banned not in label.lower()


def test_raw_text_fallback_handles_all_stopword_documents_without_crashing():
    """scikit-learn's TfidfVectorizer raises ValueError("empty vocabulary")
    when every token across all documents is filtered out as a stopword or
    probable name -- the fallback tier must degrade to "Cluster N" instead
    of propagating that exception."""
    df = pd.DataFrame(
        {
            "ticket_id": ["T1", "T2"],
            "subject": ["Hi", "Thanks"],
            "conversation_text": [
                "Guest: Hi there, thanks so much",
                "Guest: Thank you, regards",
            ],
            "cluster_id": [0, 0],
        }
    )

    labeled = assign_cluster_labels(df)
    assert labeled.iloc[0]["cluster_label"] == "Cluster 0"
