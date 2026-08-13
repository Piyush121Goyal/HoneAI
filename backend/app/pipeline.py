"""The prompt-refinement pipeline.

Four stages — analyze intent, draft, critique, refine — mirroring how a
careful engineer would iterate. The final refine stage is streamed token by
token to the client over SSE. The same steps run non-streaming inside a
Celery task for batch jobs.
"""
from __future__ import annotations

from typing import AsyncIterator

from .llm import LLMClient, Usage, build_structured_prompt, tokenize_text

STAGES = ["Understanding", "Drafting", "Refining"]

# Bump this whenever BASE_INSTRUCTIONS or any stage prompt changes. It's
# folded into the /optimize cache key so a prompt-logic change can't get
# masked by a stale cached response from before the change.
PIPELINE_VERSION = "4"

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


async def _mock_fallback(goal: str) -> AsyncIterator[dict]:
    """A clean, single-shot mock generation built from the ORIGINAL goal only.

    Used when a real provider fails partway through the multi-stage
    pipeline, before any real refine token has reached the client. Critically
    this never touches the analyze/draft/critique text already produced —
    those stages can themselves be a mix of real and mock output depending on
    where the failure hit, and feeding that accumulated blob back into the
    mock generator is exactly what used to produce garbled, duplicated,
    nested text. Restarting clean from the goal is the only safe option.
    """
    text = build_structured_prompt(goal)
    yield {"event": "stage", "data": {"index": 2}}
    for tok in tokenize_text(text):
        yield {"event": "token", "data": {"text": tok}}
    yield {
        "event": "done",
        "data": {
            "prompt": text,
            "usage": {
                "prompt_tokens": len(goal) // 4,
                "completion_tokens": len(text) // 4,
            },
        },
    }


async def optimize_stream(
    goal: str, opts: dict, llm: LLMClient | None = None
) -> AsyncIterator[dict]:
    """Yield SSE-shaped dicts: {'event': 'stage'|'token'|'done', 'data': ...}.

    Each of the 4 pipeline stages (analyze/draft/critique/refine) is its own
    LLM call, and any one of them can fail independently (rate limit, outage)
    without the others failing. If that happens before any refine token has
    been shown to the client, we restart clean on the mock generator (see
    _mock_fallback). If it happens *after* real refine tokens have already
    streamed out, we stop rather than splice in anything further — a
    real answer that ends early beats a real answer glued to a fake one.
    """
    llm = llm or LLMClient()
    total = Usage()
    refine_started = False

    try:
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
            refine_started = True
            final_parts.append(tok)
            yield {"event": "token", "data": {"text": tok}}

        final_text = "".join(final_parts)
        if not final_text.strip():
            raise RuntimeError("refine stage produced no output")

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
    except Exception:
        if refine_started:
            return
        async for evt in _mock_fallback(goal):
            yield evt


async def optimize_once(goal: str, opts: dict, llm: LLMClient | None = None) -> dict:
    """Non-streaming version — collects the full result. Used by Celery."""
    result = {"prompt": "", "usage": {}}
    async for evt in optimize_stream(goal, opts, llm):
        if evt["event"] == "done":
            result = evt["data"]
    return result
