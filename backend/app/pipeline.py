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

# Bump this whenever BASE_INSTRUCTIONS or any stage prompt changes. It's
# folded into the /optimize cache key so a prompt-logic change can't get
# masked by a stale cached response from before the change.
PIPELINE_VERSION = "3"

# Standing instructions injected into every stage of every request, regardless
# of goal/model/tone/format. This is where house rules and quality guardrails
# live — edit this list to change behavior across the whole pipeline at once.
BASE_INSTRUCTIONS = (
    "House rules, always follow these:\n"
    "- Never repeat the same word or phrase back-to-back or redundantly.\n"
    "- Never leave placeholder text like [insert X here] or TODO — write "
    "concrete, usable content or a clearly labeled example instead.\n"
    "- Use correct spacing, grammar, and punctuation throughout.\n"
    "- Do not add meta-commentary about what you are about to do or did — "
    "output only the requested content.\n"
    "- Be concise. Do not pad with filler sentences or restate the task back.\n"
    "- Always structure the final prompt using the RCTFO model: Role, Context, "
    "Task, Format, Output. The user will rarely specify all five explicitly — "
    "when a part is missing, infer the most reasonable value directly from "
    "their input (what they're trying to do, who'd plausibly ask it, what a "
    "usable result looks like) rather than leaving it out or asking a "
    "clarifying question.\n"
    "- Critical: you are never answering the user's underlying question or "
    "completing their task yourself. You are always producing a PROMPT that "
    "someone will paste into a separate AI model to get that answer later. "
    "Even for broad, personal, or advice-style goals (health, life, career, "
    "etc.), do not write the advice/guide/answer — write the Role, Context, "
    "Task, Format, and Output that would make another model produce it well."
)


def _system_for(stage: str, opts: dict) -> str:
    model = opts.get("model", "GPT-4o")
    tone = opts.get("tone", "Direct")
    fmt = opts.get("format", "Markdown")
    if stage == "analyze":
        body = (
            "You are a prompt engineer. Read the user's goal and restate, in one "
            "short paragraph, the true intent, the audience, and any implicit "
            "constraints. Do not write the prompt yet."
        )
    elif stage == "draft":
        body = (
            f"You are a prompt engineer targeting {model}. Write a first prompt "
            f"structured around the RCTFO model:\n"
            f"- Role: who/what the target model should act as.\n"
            f"- Context: the situation, audience, and any constraints.\n"
            f"- Task: exactly what to do, stated unambiguously.\n"
            f"- Format: the expected output format ({fmt}).\n"
            f"- Output: what a finished, usable result looks like.\n"
            f"Voice: {tone}."
        )
    elif stage == "critique":
        body = (
            "You are a strict reviewer. List the weaknesses of the draft prompt: "
            "vague wording, a missing or weak Role/Context/Task/Format/Output "
            "section, ambiguity, or anything that would produce an inconsistent "
            "result. Be terse."
        )
    else:
        # refine
        body = (
            f"You are a prompt engineer targeting {model}. Rewrite the draft into "
            f"the final prompt, fixing every issue raised in the critique, and "
            f"structured clearly around Role, Context, Task, Format, and Output. "
            f"Voice: {tone}. Output format for the end model: {fmt}. Return ONLY "
            f"the final prompt."
        )
    return f"{BASE_INSTRUCTIONS}\n\n{body}"


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
