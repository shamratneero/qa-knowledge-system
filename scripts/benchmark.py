"""Benchmark harness for search quality and latency.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import time
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.search.hybrid import search_hybrid


@dataclass
class EvalCase:
    query: str
    expected_id: int


EVAL_FILE = PROJECT_ROOT / "data" / "eval_set.json"


def _load_eval_set() -> list[EvalCase]:
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Eval set file not found: {EVAL_FILE}")

    raw = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        if "query" not in item or "expected_id" not in item:
            continue
        cases.append(EvalCase(query=str(item["query"]), expected_id=int(item["expected_id"])))
    return cases


def run_eval():
    eval_set = _load_eval_set()
    total = len(eval_set)
    top1 = 0
    top3 = 0
    found = 0
    true_positive = 0
    elapsed = []

    for case in eval_set:
        start = time.perf_counter()
        result = search_hybrid(case.query, top_n=3, method="hybrid")
        elapsed.append((time.perf_counter() - start) * 1000)

        if not result.get("found"):
            continue

        found += 1

        ids = [r.get("id") for r in result.get("results", [])]
        if ids and ids[0] == case.expected_id:
            top1 += 1
        if case.expected_id in ids[:3]:
            top3 += 1
            true_positive += 1

    precision = (true_positive / found) if found else 0.0
    recall = true_positive / total if total else 0.0

    sorted_elapsed = sorted(elapsed)
    p95_index = int(0.95 * (len(sorted_elapsed) - 1)) if sorted_elapsed else 0
    p95_latency = sorted_elapsed[p95_index] if sorted_elapsed else 0.0

    metrics = {
        "cases": total,
        "answered": found,
        "top1_accuracy": (top1 / total) if total else 0.0,
        "top3_accuracy": (top3 / total) if total else 0.0,
        "precision": precision,
        "recall": recall,
        "avg_latency_ms": mean(elapsed) if elapsed else 0.0,
        "p95_latency_ms": p95_latency,
    }

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_results.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Benchmark summary")
    print("-----------------")
    print(f"Cases: {metrics['cases']}")
    print(f"Answered: {metrics['answered']}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-3 accuracy: {metrics['top3_accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Avg latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"P95 latency: {metrics['p95_latency_ms']:.1f} ms")
    print(f"Saved JSON: {out_file}")


if __name__ == "__main__":
    run_eval()
