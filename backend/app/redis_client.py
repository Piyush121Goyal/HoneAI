"""Redis connection + small helpers: JSON cache and fixed-window rate limiting."""
from __future__ import annotations

import json

from redis.asyncio import Redis, from_url

from .config import settings

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = from_url(settings.redis_url, decode_responses=True)
    return _redis


async def cache_get(key: str) -> dict | None:
    raw = await get_redis().get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def cache_set(key: str, value: dict, ttl: int = 3600) -> None:
    await get_redis().set(key, json.dumps(value), ex=ttl)


async def rate_limit_hit(identity: str, limit: int, window_seconds: int) -> bool:
    """Fixed-window counter. Returns True if the caller is over the limit."""
    r = get_redis()
    key = f"rl:{identity}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, window_seconds)
    return count > limit
