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
