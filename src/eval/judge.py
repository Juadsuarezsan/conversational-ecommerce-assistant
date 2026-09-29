"""LLM-as-judge for task completion with an explicit, numbered rubric.

The judge reads a whole conversation transcript plus the final cart and answers
the rubric below with JSON. It needs ``ANTHROPIC_API_KEY``; without it
:func:`judge_conversation` raises :class:`JudgeUnavailable` and the eval report
records the cell as pending. The client is patched in tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, SecretStr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings

#: Numbered criteria. Each is scored 0 or 1 by the judge; the verdict is the sum.
RUBRIC: tuple[tuple[int, str, str], ...] = (
    (
        1,
        "task_completed",
        "The user's final goal was achieved: the requested product is in the cart with the "
        "right quantity, the refund/escalation decision was taken, or the question was "
        "answered. Example pass: user says 'add 2 of the second one' and the cart shows "
        "quantity 2 of the product that was second in the previous list.",
    ),
    (
        2,
        "grounded_citations",
        "Every product mentioned in the assistant's replies appears in the retrieved list or "
        "the cart and is cited as [pid:N]. Example fail: the assistant recommends a product "
        "that was never retrieved.",
    ),
    (
        3,
        "context_tracking",
        "References to earlier turns ('that one', 'the ketchup', 'make it 3') were resolved "
        "to the product the user meant. Example fail: 'remove the almond milk' removes a "
        "different milk.",
    ),
    (
        4,
        "policy_compliance",
        "Refund and escalation policy was applied: refunds over $100 or in "
        "alcohol/tobacco/pharmacy/babies were escalated; smaller, non-sensitive refunds were "
        "auto-approved; explicit requests for a human were escalated.",
    ),
    (
        5,
        "no_hallucinated_actions",
        "The assistant never claims an action it did not perform (nothing 'added' that is "
        "missing from the cart snapshot, no invented order numbers or prices).",
    ),
)

SYSTEM_PROMPT = (
    "You are a strict evaluator of a shopping assistant. Score the conversation against "
    "each numbered criterion with 0 (fail) or 1 (pass). Use only the transcript and the "
    "final cart. Return JSON only:\n"
    '{"scores": {"1": 0|1, "2": 0|1, "3": 0|1, "4": 0|1, "5": 0|1}, "rationale": "..."}\n\n'
    "Criteria:\n" + "\n".join(f"{n}. {name}: {desc}" for n, name, desc in RUBRIC)
)


class JudgeUnavailable(RuntimeError):
    """Raised when the judge cannot run (no API key)."""


class JudgeVerdict(BaseModel):
    """Structured output of the judge."""

    scores: dict[str, int] = Field(default_factory=dict)
    rationale: str = ""

    @property
    def task_completed(self) -> bool:
        """Criterion 1."""
        return self.scores.get("1", 0) == 1

    @property
    def total(self) -> int:
        """Sum of the five criteria (0-5)."""
        return sum(int(v) for v in self.scores.values())


def format_transcript(turns: Sequence[dict[str, Any]], cart: Sequence[tuple[int, int]]) -> str:
    """Render turns and the final cart for the judge prompt."""
    lines = []
    for i, t in enumerate(turns, 1):
        lines.append(f"[{i}] USER: {t['user']}")
        lines.append(f"[{i}] ASSISTANT ({t.get('intent', '?')}): {t['assistant']}")
    lines.append("FINAL CART: " + (", ".join(f"pid {p} x{q}" for p, q in cart) or "empty"))
    return "\n".join(lines)


class LLMJudge:
    """Claude-backed judge; ``_build_chat`` is patched in tests."""

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 30.0):
        s = get_settings()
        self.model = model or s.anthropic_model
        self.api_key = api_key if api_key is not None else s.anthropic_api_key
        self.timeout = timeout

    def _build_chat(self) -> Any:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model,
            api_key=SecretStr(self.api_key or ""),
            temperature=0.0,
            max_tokens=400,
            timeout=self.timeout,
            max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((ValueError, TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def judge(self, transcript: str) -> JudgeVerdict:
        """Score one transcript; raises :class:`JudgeUnavailable` without a key."""
        if not self.api_key:
            raise JudgeUnavailable("LLM judge requires ANTHROPIC_API_KEY")
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = self._build_chat()
        resp = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=transcript)]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            return JudgeVerdict.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("judge returned invalid JSON") from exc


async def judge_conversation(
    outcome: dict[str, Any], judge: LLMJudge | None = None
) -> JudgeVerdict:
    """Judge one outcome produced by :mod:`src.eval.conversation`."""
    j = judge or LLMJudge()
    final_cart = outcome["turns"][-1]["cart"] if outcome["turns"] else []
    return await j.judge(format_transcript(outcome["turns"], final_cart))
