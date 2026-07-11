from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app


def run_api_tests():
    client = TestClient(app)

    print("Health check:")
    r = client.get("/health")
    print(r.status_code, r.json())

    queries = ["AI", "Machine Learning", "nonexistentterm"]
    for q in queries:
        print(f"\nQuery: {q}")
        r = client.get("/search", params={"q": q})
        print(r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)


if __name__ == "__main__":
    run_api_tests()
