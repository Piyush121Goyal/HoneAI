"""The main interactive flow: stream an optimized prompt over SSE."""
import hashlib
import json

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from ..database import get_db
from ..deps import enforce_rate_limit, get_optional_user
from ..llm import LLMClient
from ..models import UsageLog, User
from ..pipeline import PIPELINE_VERSION, optimize_stream
from ..redis_client import cache_get, cache_set
from ..schemas import OptimizeIn

router = APIRouter(tags=["optimize"])


@router.post("/optimize", dependencies=[Depends(enforce_rate_limit)])
async def optimize(
    body: OptimizeIn,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    opts = {"model": body.model, "tone": body.tone, "format": body.format}
    cache_key = "opt:" + hashlib.sha256(
        json.dumps(
            {"v": PIPELINE_VERSION, "g": body.goal, **opts}, sort_keys=True
        ).encode()
    ).hexdigest()

    async def event_source():
        llm = LLMClient()

        # Serve cached results instantly (replayed as a stream for a smooth UX).
        cached = await cache_get(cache_key)
        if cached:
            yield {"event": "stage", "data": json.dumps({"index": 2})}
            for chunk in _chunks(cached["prompt"], 24):
                yield {"event": "token", "data": json.dumps({"text": chunk})}
            yield {"event": "done", "data": json.dumps(cached)}
            return

        final = {}
        async for evt in optimize_stream(body.goal, opts, llm):
            if evt["event"] == "done":
                final = evt["data"]
            yield {"event": evt["event"], "data": json.dumps(evt["data"])}

        if final.get("prompt"):
            await cache_set(cache_key, final, ttl=86400)
            usage = final.get("usage", {})
            db.add(
                UsageLog(
                    user_id=user.id if user else None,
                    model=body.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost_usd=0.0,
                )
            )
            await db.commit()

    return EventSourceResponse(event_source())


def _chunks(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i : i + n]
