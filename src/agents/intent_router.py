"""Intent classifier — Claude with structured output (Pydantic schema enforced)."""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import IntentDecision

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


class IntentRouter:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def classify(self, message: str, history: list[dict[str, str]] | None = None) -> IntentDecision:
        # Offline deterministic fallback (no API key) — keeps tests + CI usable
        if not self.api_key:
            return self._heuristic(message)

        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = ChatAnthropic(
            model=self.model, api_key=self.api_key, temperature=0.0, max_tokens=400, timeout=15.0,
        )
        history_text = ""
        if history:
            history_text = "\n".join(f"[{h['role']}]: {h['content']}" for h in history[-6:])
        user_block = (
            (f"<conversation>\n{history_text}\n</conversation>\n" if history_text else "")
            + f"<message>{message}</message>"
        )
        resp = await chat.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_block)])
        text = self._strip_fences(resp.content if isinstance(resp.content, str) else str(resp.content))
        raw = json.loads(text)
        return IntentDecision(**raw)

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.lower().startswith("json"):
                t = t[4:].lstrip()
        return t

    @staticmethod
    def _heuristic(message: str) -> IntentDecision:
        m = message.lower()
        if any(w in m for w in ("refund", "return", "money back")):
            return IntentDecision(intent="refund", confidence=0.7, rationale="keyword: refund/return")
        if any(w in m for w in ("cart", "add to", "remove", "checkout")):
            return IntentDecision(intent="cart_op", confidence=0.7, rationale="keyword: cart")
        if any(w in m for w in ("order", "tracking", "delivery", "shipped")):
            return IntentDecision(intent="order_status", confidence=0.7, rationale="keyword: order")
        if any(w in m for w in ("human", "agent", "person", "escalate")):
            return IntentDecision(intent="escalate", confidence=0.9, rationale="explicit request")
        if m.strip() in ("hi", "hello", "hey", "thanks", "thank you", "bye"):
            return IntentDecision(intent="greeting", confidence=0.9, rationale="small talk")
        return IntentDecision(intent="product_search", confidence=0.55, rationale="default")
