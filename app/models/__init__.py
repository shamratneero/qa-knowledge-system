from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


__all__ = ["QuestionRequest"]