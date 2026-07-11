"""Simple benchmark harness for search quality and latency.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.search.hybrid import search_hybrid


@dataclass
class EvalCase:
    query: str
    expected_id: int


EVAL_SET = [
    EvalCase("What is AI?", 1),
    EvalCase("Artificial intelligence basics", 1),
    EvalCase("What is machine learning?", 2),
    EvalCase("Machne Lerning", 2),
    EvalCase("ML patterns from data", 2),
]


def run_eval():
    total = len(EVAL_SET)
    top1 = 0
    top3 = 0
    elapsed = []

    for case in EVAL_SET:
        start = time.perf_counter()
        result = search_hybrid(case.query, top_n=3, method="hybrid")
        elapsed.append((time.perf_counter() - start) * 1000)

        if not result.get("found"):
            continue

        ids = [r.get("id") for r in result.get("results", [])]
        if ids and ids[0] == case.expected_id:
            top1 += 1
        if case.expected_id in ids[:3]:
            top3 += 1

    print("Benchmark summary")
    print("-----------------")
    print(f"Cases: {total}")
    print(f"Top-1 accuracy: {top1 / total:.2%}")
    print(f"Top-3 accuracy: {top3 / total:.2%}")
    print(f"Avg latency: {sum(elapsed) / len(elapsed):.1f} ms")


if __name__ == "__main__":
    run_eval()
