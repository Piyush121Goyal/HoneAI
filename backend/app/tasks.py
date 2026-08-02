"""Celery tasks: embedding backfill for saved prompts."""
import anyio
from sqlalchemy import select

from .celery_app import celery
from .database import SessionLocal
from .llm import LLMClient
from .models import Prompt


async def _embed_prompt(prompt_id: str, goal: str) -> None:
    embedding = await LLMClient().embed(goal)
    async with SessionLocal() as db:
        prompt = await db.get(Prompt, prompt_id)
        if prompt:
            prompt.embedding = embedding
            await db.commit()


@celery.task(name="app.tasks.embed_prompt")
def embed_prompt(prompt_id: str, goal: str) -> None:
    anyio.run(_embed_prompt, prompt_id, goal)


async def _backfill_embeddings() -> int:
    async with SessionLocal() as db:
        rows = await db.scalars(select(Prompt).where(Prompt.embedding.is_(None)))
        prompts = list(rows)
        for p in prompts:
            p.embedding = await LLMClient().embed(p.goal)
        await db.commit()
        return len(prompts)


@celery.task(name="app.tasks.backfill_embeddings")
def backfill_embeddings() -> int:
    return anyio.run(_backfill_embeddings)
