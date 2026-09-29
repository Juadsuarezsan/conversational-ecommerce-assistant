"""Intent Router: Claude with a strict JSON schema, or a deterministic keyword fallback.

The LLM path is used only when an ``ANTHROPIC_API_KEY`` is configured. Every call
has an explicit timeout and exponential-backoff retries; malformed JSON from the
model is retried, then surfaces as ``ValueError``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import SecretStr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.schemas import IntentDecision, Usage
from src.observability import usage_from_response

SYSTEM_PROMPT = """You are the Intent Router of a conversational e-commerce assistant.
Classify the user message into ONE of these intents and return JSON only (no prose):

- product_search   — user wants to find products ("show me healthy breakfast cereals")
- cart_op          — add/remove/update/view cart ("add 2 to cart", "what's in my cart")
- order_status     — checking on existing orders ("where is my order #1234")
- refund           — return / refund a previous purchase
- escalate         — explicit ask for human, OR profanity, OR sensitive/legal/medical topic
- greeting         — hello/thank-you/small-talk only, no other intent

Also extract any obvious filters mentioned (price, category, dietary needs).

Strict JSON schema:
{
  "intent": "<one of the values above>",
  "confidence": 0.0-1.0,
  "extracted_filters": { "price_max": ..., "category": ..., "diet": "vegan|gluten_free|lactose_free|kosher|..." },
  "rationale": "one-sentence explanation"
}
"""

_REFUND = ("refund", "return the", "return my", "money back", "send it back", "give me back")
_CART = (
    "cart",
    "basket",
    "add ",
    "add two",
    "add one",
    "remove",
    "checkout",
    "check out",
    "swap that",
    "replace that",
    "take out",
    "quantity",
    "qty",
    "change the",
    "empty my",
    "clear my",
    "make that",
    "make it",
    "make them",
)
_ORDER = ("order #", "my order", "order status", "tracking", "delivery", "shipped", "where is my")
_ESCALATE = (
    "human",
    "agent",
    "real person",
    "representative",
    "manager",
    "supervisor",
    "escalate",
    "unacceptable",
    "lawyer",
    "complaint",
    "sue ",
)
_GREETING = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "good morning",
    "hi there",
    "thanks!",
}


class IntentRouter:
    """Classify a message into an :class:`IntentDecision`.

    Args:
        model: Pinned Anthropic model id.
        api_key: Anthropic key; ``None`` selects the heuristic fallback.
        timeout: Seconds before an LLM call is abandoned.
    """

    def __init__(self, model: str, api_key: str | None, timeout: float = 15.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @property
    def llm_enabled(self) -> bool:
        """Whether the LLM path is active."""
        return bool(self.api_key)

    def _build_chat(self) -> Any:
        """Instantiate the LangChain Anthropic client (patched in tests)."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model,
            api_key=SecretStr(self.api_key or ""),
            temperature=0.0,
            max_tokens=400,
            timeout=self.timeout,
            max_retries=0,
        )

    async def classify(
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> tuple[IntentDecision, Usage]:
        """Return the decision and the token usage (zero for the heuristic path)."""
        if not self.api_key:
            return self._heuristic(message), Usage()
        return await self._classify_llm(message, history or [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ValueError, TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def _classify_llm(
        self, message: str, history: list[dict[str, str]]
    ) -> tuple[IntentDecision, Usage]:
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = self._build_chat()
        history_text = "\n".join(f"[{h['role']}]: {h['content']}" for h in history[-6:])
        user_block = (
            f"<conversation>\n{history_text}\n</conversation>\n" if history_text else ""
        ) + f"<message>{message}</message>"
        resp = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_block)]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        try:
            raw = json.loads(self._strip_fences(text))
            decision = IntentDecision.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Intent router returned invalid JSON: {}", text[:200])
            raise ValueError("intent router returned invalid JSON") from exc
        return decision, usage_from_response(resp)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json fences if the model added them."""
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        return t.strip()

    @staticmethod
    def _heuristic(message: str) -> IntentDecision:
        """Keyword classifier used when no API key is configured (deterministic)."""
        m = f" {message.lower().strip()} "
        if m.strip() in _GREETING:
            return IntentDecision(intent="greeting", confidence=0.9, rationale="small talk")
        if any(w in m for w in _ESCALATE):
            return IntentDecision(intent="escalate", confidence=0.9, rationale="explicit request")
        if any(w in m for w in _REFUND):
            return IntentDecision(
                intent="refund", confidence=0.7, rationale="keyword: refund/return"
            )
        if any(w in m for w in _CART):
            return IntentDecision(intent="cart_op", confidence=0.7, rationale="keyword: cart")
        if any(w in m for w in _ORDER):
            return IntentDecision(intent="order_status", confidence=0.7, rationale="keyword: order")
        return IntentDecision(intent="product_search", confidence=0.55, rationale="default")
