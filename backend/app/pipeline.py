"""The prompt-refinement pipeline.

Four stages — analyze intent, draft, critique, refine — mirroring how a
careful engineer would iterate. The final refine stage is streamed token by
token to the client over SSE. The same steps run non-streaming inside a
Celery task for batch jobs.
"""
from __future__ import annotations

from typing import AsyncIterator

from .llm import LLMClient, Usage

STAGES = ["Understanding", "Drafting", "Refining"]


def _system_for(stage: str, opts: dict) -> str:
    model = opts.get("model", "GPT-4o")
    tone = opts.get("tone", "Direct")
    fmt = opts.get("format", "Markdown")
    if stage == "analyze":
        return (
            "You are a prompt engineer. Read the user's goal and restate, in one "
            "short paragraph, the true intent, the audience, and any implicit "
            "constraints. Do not write the prompt yet."
        )
    if stage == "draft":
        return (
            f"You are a prompt engineer targeting {model}. Write a first "
            f"structured prompt with a role, task, context, requirements, an "
            f"explicit output format ({fmt}), and guardrails. Voice: {tone}."
        )
    if stage == "critique":
        return (
            "You are a strict reviewer. List the weaknesses of the draft prompt: "
            "vague wording, missing constraints, ambiguity, or anything that would "
            "produce an inconsistent result. Be terse."
        )
    # refine
    return (
        f"You are a prompt engineer targeting {model}. Rewrite the draft into the "
        f"final prompt, fixing every issue raised in the critique. Voice: {tone}. "
        f"Output format for the end model: {fmt}. Return ONLY the final prompt."
    )


async def optimize_stream(
    goal: str, opts: dict, llm: LLMClient | None = None
) -> AsyncIterator[dict]:
    """Yield SSE-shaped dicts: {'event': 'stage'|'token'|'done', 'data': ...}."""
    llm = llm or LLMClient()
    total = Usage()

    # stage 0 — understand
    yield {"event": "stage", "data": {"index": 0}}
    intent, u = await llm.complete(_system_for("analyze", opts), goal)
    total.prompt_tokens += u.prompt_tokens
    total.completion_tokens += u.completion_tokens

    # stage 1 — draft
    yield {"event": "stage", "data": {"index": 1}}
    draft, u = await llm.complete(
        _system_for("draft", opts), f"Goal: {goal}\n\nIntent analysis:\n{intent}"
    )
    total.prompt_tokens += u.prompt_tokens
    total.completion_tokens += u.completion_tokens

    # (critique runs internally to guide the refine step)
    critique, u = await llm.complete(_system_for("critique", opts), draft)
    total.prompt_tokens += u.prompt_tokens
    total.completion_tokens += u.completion_tokens

    # stage 2 — refine (streamed)
    yield {"event": "stage", "data": {"index": 2}}
    refine_input = (
        f"Goal: {goal}\n\nDraft:\n{draft}\n\nCritique to fix:\n{critique}"
    )
    final_parts: list[str] = []
    async for tok in llm.stream(_system_for("refine", opts), refine_input):
        if not tok:
            continue
        final_parts.append(tok)
        yield {"event": "token", "data": {"text": tok}}

    final_text = "".join(final_parts)
    total.completion_tokens += len(final_text) // 4
    yield {
        "event": "done",
        "data": {
            "prompt": final_text,
            "usage": {
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
            },
        },
    }


async def optimize_once(goal: str, opts: dict, llm: LLMClient | None = None) -> dict:
    """Non-streaming version — collects the full result. Used by Celery."""
    result = {"prompt": "", "usage": {}}
    async for evt in optimize_stream(goal, opts, llm):
        if evt["event"] == "done":
            result = evt["data"]
    return result
