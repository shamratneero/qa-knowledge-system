from __future__ import annotations

from app.search import search


def run_manual_tests():
    queries = ["AI", "Machine Learning", "artificial intelligence", "nonexistentterm"]
    for q in queries:
        print("\n=== QUERY: %s ===" % q)
        res = search(q)
        print(res)


if __name__ == "__main__":
    run_manual_tests()
