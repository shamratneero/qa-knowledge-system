"""FastAPI entrypoint for QA Knowledge System."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import QuestionRequest
from app.search import search


app = FastAPI(
    title="Q&A Knowledge System",
    description="Excel-powered Q&A API",
    version="1.0.0",
)


# Allow basic CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Q&A Knowledge System API is running!"}


@app.post(
    "/ask",
    tags=["Knowledge Base"],
    summary="Search the knowledge base",
)
def ask_question(request: QuestionRequest):
    result = search(request.question)
    if not result.get("found"):
        # return standard not-found payload with 404
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


__all__ = ["app", "ask_question", "home"]
