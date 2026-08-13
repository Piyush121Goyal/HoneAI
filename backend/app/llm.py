"""Async LLM access via httpx.

Supports Anthropic and OpenAI, and falls back to a deterministic *mock*
provider so the whole app runs end-to-end without any API key. Every method
is non-blocking so one worker can serve many concurrent users.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .config import settings


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMClient:
    provider: str = field(default_factory=lambda: settings.llm_provider)

    # ------------------------- public methods ------------------------- #
    async def complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        """Single-shot, non-streaming completion (used by pipeline stages).

        Falls back to the mock provider on any upstream failure (bad key,
        quota exhausted, provider outage) so a broken API key degrades the
        output instead of breaking the app.
        """
        try:
            if self.provider == "anthropic" and settings.anthropic_api_key:
                return await self._anthropic_complete(system, prompt)
            if self.provider == "openai" and settings.openai_api_key:
                return await self._openai_complete(system, prompt)
            if self.provider == "gemini" and settings.gemini_api_key:
                return await self._gemini_complete(system, prompt)
            if self.provider == "groq" and settings.groq_api_key:
                return await self._groq_complete(system, prompt)
        except (httpx.HTTPError, KeyError, IndexError):
            pass
        return self._mock_complete(system, prompt)

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        """Streaming completion (used by the final refine stage / SSE).

        Falls back to the mock provider only if the real provider fails
        *before* producing any output. If it fails partway through (a
        connection drop mid-stream, say), we stop instead of splicing in a
        fresh, unrelated mock completion — half a real answer glued to a
        whole fake one is worse than a real answer that ends early.
        """
        streamer = None
        if self.provider == "anthropic" and settings.anthropic_api_key:
            streamer = self._anthropic_stream(system, prompt)
        elif self.provider == "openai" and settings.openai_api_key:
            streamer = self._openai_stream(system, prompt)
        elif self.provider == "gemini" and settings.gemini_api_key:
            streamer = self._gemini_stream(system, prompt)
        elif self.provider == "groq" and settings.groq_api_key:
            streamer = self._groq_stream(system, prompt)

        if streamer is None:
            for tok in _tokenize(self._mock_complete(system, prompt)[0]):
                yield tok
            return

        yielded_any = False
        try:
            async for tok in streamer:
                yielded_any = True
                yield tok
        except (httpx.HTTPError, KeyError, IndexError):
            if not yielded_any:
                for tok in _tokenize(self._mock_complete(system, prompt)[0]):
                    yield tok

    async def embed(self, text: str) -> list[float]:
        if self.provider == "openai" and settings.openai_api_key:
            return await self._openai_embed(text)
        return _mock_embedding(text, settings.embed_dim)

    # ------------------------------ anthropic ------------------------------ #
    async def _anthropic_complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []))
            u = data.get("usage", {})
            return text, Usage(u.get("input_tokens", 0), u.get("output_tokens", 0))

    async def _anthropic_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "stream": True,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        evt = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        yield evt["delta"].get("text", "")

    # ------------------------------- openai ------------------------------- #
    async def _openai_complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "frequency_penalty": 0.3,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {})
            return text, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    async def _openai_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "frequency_penalty": 0.3,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                        yield evt["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ------------------------------- gemini ------------------------------- #
    async def _gemini_complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent",
                params={"key": settings.gemini_api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1500},
                },
            )
            r.raise_for_status()
            data = r.json()
            parts = data["candidates"][0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            u = data.get("usageMetadata", {})
            return text, Usage(
                u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
            )

    async def _gemini_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:streamGenerateContent",
                params={"key": settings.gemini_api_key, "alt": "sse"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1500},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        evt = json.loads(payload)
                        parts = evt["candidates"][0]["content"].get("parts", [])
                        for part in parts:
                            if part.get("text"):
                                yield part["text"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # -------------------------------- groq --------------------------------- #
    # Groq's API is OpenAI-compatible (chat completions wire format), just a
    # different base URL and key.
    async def _groq_complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "frequency_penalty": 0.3,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            u = data.get("usage", {})
            return text, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    async def _groq_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": settings.llm_model,
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "frequency_penalty": 0.3,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                        yield evt["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def _openai_embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": "text-embedding-3-small", "input": text},
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]

    # -------------------------------- mock -------------------------------- #
    def _mock_complete(self, system: str, prompt: str) -> tuple[str, Usage]:
        text = build_structured_prompt(prompt)
        return text, Usage(len(prompt) // 4, len(text) // 4, 0.0)


# --------------------------- helper functions --------------------------- #
def _tokenize(s: str) -> list[str]:
    return re.findall(r"\s+|\S+", s)


def _mock_embedding(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding so semantic search runs without a key."""
    vec: list[float] = []
    seed = text.encode("utf-8")
    i = 0
    while len(vec) < dim:
        h = hashlib.sha256(seed + str(i).encode()).digest()
        for b in h:
            vec.append((b / 255.0) * 2 - 1)
            if len(vec) >= dim:
                break
        i += 1
    # normalize
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def build_structured_prompt(goal: str) -> str:
    """Same structuring logic the frontend previews, used by the mock LLM."""
    g = " ".join(goal.split()).rstrip(". ")
    g = (g[:1].upper() + g[1:]) if g else "Explain a topic clearly to a beginner"
    role = _infer_role(g.lower())
    return (
        f"You are {role}. You are careful, concrete, and never pad your answers.\n\n"
        f"## Task\n{g}.\n\n"
        "## Context\n"
        "- The reader wants a result they can use immediately, not a lecture.\n"
        "- Prefer real examples over abstract description.\n\n"
        "## Requirements\n"
        "- Lead with the answer; put reasoning after, only if it helps.\n"
        "- Be specific: use names, numbers, and concrete examples.\n"
        "- If a key detail is missing, ask exactly one clarifying question first.\n\n"
        "## Output format\n"
        "Return clean Markdown. Use short paragraphs and headings where helpful.\n\n"
        "## Guardrails\n"
        "- Do not invent facts, sources, or statistics.\n"
        "- If you are unsure, say so plainly instead of guessing.\n"
        "- Stop when the task is done — no summary of what you just wrote."
    )


def _infer_role(s: str) -> str:
    if re.search(r"code|function|bug|api|python|javascript|sql|regex|program", s):
        return "a senior software engineer who writes production-quality code"
    if re.search(r"email|reply|message|outreach|cold|linkedin", s):
        return "a sharp communications lead who writes messages people reply to"
    if re.search(r"market|ad|copy|landing|brand|slogan|tagline", s):
        return "a marketing strategist with a copywriter's instincts"
    if re.search(r"essay|blog|article|story|write|draft|post", s):
        return "an experienced editor with a clear, unfussy writing style"
    if re.search(r"plan|strategy|roadmap|business|analyze|research", s):
        return "an analyst who turns messy questions into decisions"
    if re.search(r"teach|explain|learn|beginner|understand", s):
        return "a patient teacher who explains things in plain language"
    return "a domain expert who gives direct, well-organized answers"
