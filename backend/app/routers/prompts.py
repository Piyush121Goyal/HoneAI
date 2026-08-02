"""Prompt library: save, list, retrieve, rate, and semantically search."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..deps import get_current_user
from ..llm import LLMClient
from ..models import Feedback, Prompt, PromptVersion, User
from ..schemas import (
    FeedbackIn,
    PromptCreate,
    PromptDetailOut,
    PromptOut,
    SearchIn,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("", response_model=list[PromptOut])
async def list_prompts(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    rows = await db.scalars(
        select(Prompt).where(Prompt.user_id == user.id).order_by(Prompt.created_at.desc())
    )
    return list(rows)


@router.post("", response_model=PromptOut, status_code=201)
async def create_prompt(
    body: PromptCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = Prompt(
        user_id=user.id,
        goal=body.goal,
        model=body.model,
        tone=body.tone,
        format=body.format,
        rating=body.rating,
        latest_version=1,
    )
    db.add(prompt)
    await db.flush()  # assigns prompt.id
    db.add(
        PromptVersion(
            prompt_id=prompt.id, version=1, content=body.prompt, meta=body.meta
        )
    )
    await db.commit()
    await db.refresh(prompt)

    # Compute the embedding in the background (Celery + Redis broker).
    try:
        from ..tasks import embed_prompt

        embed_prompt.delay(prompt.id, prompt.goal)
    except Exception:
        pass  # worker offline; embedding can be backfilled later

    return prompt


@router.get("/search", response_model=list[PromptOut])
async def search_prompts(
    body: SearchIn = Depends(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Semantic search over saved prompts using pgvector cosine distance."""
    q_embedding = await LLMClient().embed(body.query)
    rows = await db.scalars(
        select(Prompt)
        .where(Prompt.user_id == user.id, Prompt.embedding.isnot(None))
        .order_by(Prompt.embedding.cosine_distance(q_embedding))
        .limit(body.k)
    )
    return list(rows)


@router.get("/{prompt_id}", response_model=PromptDetailOut)
async def get_prompt(
    prompt_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = await db.get(Prompt, prompt_id)
    if not prompt or prompt.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    await db.refresh(prompt, ["versions"])
    return prompt


@router.post("/{prompt_id}/feedback", status_code=201)
async def rate_prompt(
    prompt_id: str,
    body: FeedbackIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = await db.get(Prompt, prompt_id)
    if not prompt or prompt.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    db.add(
        Feedback(
            prompt_id=prompt_id, user_id=user.id, rating=body.rating, comment=body.comment
        )
    )
    prompt.rating = 1 if body.rating >= 4 else -1 if 0 < body.rating < 3 else prompt.rating
    await db.commit()
    return {"ok": True}


@router.delete("/{prompt_id}", status_code=204)
async def delete_prompt(
    prompt_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = await db.get(Prompt, prompt_id)
    if not prompt or prompt.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prompt not found")
    await db.execute(delete(Prompt).where(Prompt.id == prompt_id))
    await db.commit()
