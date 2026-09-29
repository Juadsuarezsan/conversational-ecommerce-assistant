"""Response Synthesizer: Claude writes the reply with ``[pid:N]`` citations, or a template does.

The template fallback is deterministic and is what the API returns when no
``ANTHROPIC_API_KEY`` is configured (CI, offline demo, red-team target).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.schemas import CartItem, RetrievedProduct, Usage
from src.observability import usage_from_response

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
    """Compose the final answer from intent, products, cart and notes."""

    def __init__(self, model: str, api_key: str | None, timeout: float = 15.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _build_chat(self) -> Any:
        """Instantiate the LangChain Anthropic client (patched in tests)."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model,
            api_key=SecretStr(self.api_key or ""),
            temperature=0.4,
            max_tokens=600,
            timeout=self.timeout,
            max_retries=0,
        )

    async def respond(
        self,
        *,
        message: str,
        intent: str,
        products: Sequence[RetrievedProduct] = (),
        cart: Sequence[CartItem] = (),
        escalated: bool = False,
        note: str = "",
    ) -> tuple[str, Usage]:
        """Return the reply text and token usage (zero for the template path)."""
        if not self.api_key:
            return self.template(message, intent, products, cart, escalated, note), Usage()
        return await self._respond_llm(message, intent, products, cart, escalated, note)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def _respond_llm(
        self,
        message: str,
        intent: str,
        products: Sequence[RetrievedProduct],
        cart: Sequence[CartItem],
        escalated: bool,
        note: str,
    ) -> tuple[str, Usage]:
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = self._build_chat()
        ctx = [f"<message>{message}</message>", f"<intent>{intent}</intent>"]
        if products:
            ctx.append("<products>")
            for p in products:
                price = f"${p.price_usd:.2f}" if p.price_usd is not None else "—"
                ctx.append(
                    f"  [pid:{p.product_id}] {p.product_name} · {p.aisle} · {price} · score {p.score:.3f}"
                )
            ctx.append("</products>")
        if cart:
            ctx.append("<cart>")
            for ci in cart:
                ctx.append(
                    f"  [pid:{ci.product_id}] {ci.product_name} x {ci.quantity} @ ${ci.unit_price_usd:.2f}"
                )
            ctx.append("</cart>")
        if note:
            ctx.append(f"<system_note>{note}</system_note>")
        if escalated:
            ctx.append("<note>This conversation has been flagged for human escalation.</note>")
        resp = await chat.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(ctx))]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return text, usage_from_response(resp)

    @staticmethod
    def template(
        message: str,
        intent: str,
        products: Sequence[RetrievedProduct],
        cart: Sequence[CartItem],
        escalated: bool,
        note: str = "",
    ) -> str:
        """Deterministic reply used without an API key. Always cites ``[pid:N]``."""
        if escalated:
            reason = f" ({note})" if note else ""
            return f"I'm escalating this to a human teammate{reason} — they'll follow up shortly."
        if intent == "greeting":
            return "Hi! What can I help you find today?"
        if intent in ("cart_op", "refund", "order_status"):
            parts = [note] if note else []
            if cart:
                total = sum(c.unit_price_usd * c.quantity for c in cart)
                lines = "; ".join(
                    f"[pid:{c.product_id}] {c.product_name} x {c.quantity}" for c in cart
                )
                parts.append(f"Cart: {lines}. Total ${total:.2f}.")
            elif intent == "cart_op":
                parts.append("Your cart is empty.")
            return " ".join(parts) if parts else "Done."
        if products:
            top = products[:3]
            lines = ", ".join(
                f"[pid:{p.product_id}] {p.product_name}"
                + (f" (${p.price_usd:.2f})" if p.price_usd is not None else "")
                for p in top
            )
            prefix = f"{note} " if note else ""
            return f"{prefix}Top matches for your query: {lines}."
        return "I couldn't find anything for that query — could you rephrase?"
