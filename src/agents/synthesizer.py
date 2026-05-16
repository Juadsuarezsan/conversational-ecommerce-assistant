"""Response synthesizer — Claude takes retrieved products + intent and writes the reply."""
from __future__ import annotations

from typing import Sequence

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import CartItem, RetrievedProduct

SYSTEM_PROMPT = """You are the response synthesizer of a friendly e-commerce assistant.

Rules:
- Cite product IDs as [pid:NNN] inline whenever you mention a product. Required for traceability.
- Keep responses to 3-5 sentences for product results, 1-2 for greetings.
- Never invent products that are not in the provided list.
- If the cart is shown, summarize total + items briefly.
- If the user is being escalated, acknowledge calmly and confirm a human will follow up.
- Match the user's language (Spanish or English).
"""


class Synthesizer:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    async def respond(
        self,
        *,
        message: str,
        intent: str,
        products: Sequence[RetrievedProduct] = (),
        cart: Sequence[CartItem] = (),
        escalated: bool = False,
    ) -> str:
        if not self.api_key:
            return self._template_fallback(message, intent, products, cart, escalated)

        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = ChatAnthropic(
            model=self.model, api_key=self.api_key, temperature=0.4, max_tokens=600, timeout=15.0,
        )
        ctx_lines = [f"<message>{message}</message>", f"<intent>{intent}</intent>"]
        if products:
            ctx_lines.append("<products>")
            for p in products:
                price = f"${p.price_usd:.2f}" if p.price_usd else "—"
                ctx_lines.append(f"  [pid:{p.product_id}] {p.product_name} · {p.aisle} · {price} · score {p.score:.3f}")
            ctx_lines.append("</products>")
        if cart:
            ctx_lines.append("<cart>")
            for ci in cart:
                ctx_lines.append(f"  [pid:{ci.product_id}] {ci.product_name} × {ci.quantity} @ ${ci.unit_price_usd:.2f}")
            ctx_lines.append("</cart>")
        if escalated:
            ctx_lines.append("<note>This conversation has been flagged for human escalation.</note>")
        user_block = "\n".join(ctx_lines)

        resp = await chat.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_block)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    @staticmethod
    def _template_fallback(
        message: str,
        intent: str,
        products: Sequence[RetrievedProduct],
        cart: Sequence[CartItem],
        escalated: bool,
    ) -> str:
        if escalated:
            return "I'm escalating this to a human teammate — they'll follow up shortly."
        if intent == "greeting":
            return "Hi! What can I help you find today?"
        if intent == "cart_op" and cart:
            total = sum(c.unit_price_usd * c.quantity for c in cart)
            return f"Cart has {len(cart)} item(s), total ${total:.2f}."
        if products:
            top = products[:3]
            lines = ", ".join(f"[pid:{p.product_id}] {p.product_name}" for p in top)
            return f"Top matches for your query: {lines}."
        return "I couldn't find anything for that query — could you rephrase?"
