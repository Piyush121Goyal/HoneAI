"""Pydantic request/response schemas."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ------------------------------- auth ------------------------------- #
class SignupIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class GoogleIn(BaseModel):
    id_token: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    email: EmailStr
    provider: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ----------------------------- optimize ----------------------------- #
class OptimizeIn(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    model: str = "GPT-4o"
    tone: str = "Direct"
    format: str = "Markdown"


# ----------------------------- prompts ------------------------------ #
class PromptCreate(BaseModel):
    goal: str
    model: str = "GPT-4o"
    tone: str = "Direct"
    format: str = "Markdown"
    prompt: str
    rating: int = 0
    meta: dict = Field(default_factory=dict)


class PromptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    goal: str
    model: str
    tone: str
    format: str
    rating: int
    latest_version: int
    created_at: datetime
    updated_at: datetime


class PromptVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    content: str
    meta: dict
    created_at: datetime


class PromptDetailOut(PromptOut):
    versions: list[PromptVersionOut] = Field(default_factory=list)


class SearchIn(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)


class FeedbackIn(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
