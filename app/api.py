from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from .search import search


app = FastAPI(title="QA Knowledge Search")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search_endpoint(q: str = Query(..., min_length=1), top_n: int = Query(5, ge=1, le=50)):
    """Search endpoint that forwards to app.search.search()."""
    result = search(q, top_n=top_n)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


__all__ = ["app", "search_endpoint"]
