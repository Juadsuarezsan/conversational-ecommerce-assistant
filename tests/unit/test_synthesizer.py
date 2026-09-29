"""Synthesizer: template fallback and the mocked Claude path."""

from __future__ import annotations

from typing import Any

from src.agents.synthesizer import Synthesizer
from src.api.schemas import CartItem, RetrievedProduct


def _rp(pid: int, name: str, price: float | None = 2.0) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=pid, product_name=name, aisle="a", department="d", price_usd=price, score=0.5
    )


CART = [CartItem(product_id=1, product_name="Ketchup", quantity=2, unit_price_usd=4.29)]


async def test_template_cites_pids_and_prices() -> None:
    synth = Synthesizer(model="m", api_key=None)
    text, usage = await synth.respond(
        message="ketchup",
        intent="product_search",
        products=[_rp(1, "Heinz"), _rp(2, "Other", None)],
    )
    assert "[pid:1] Heinz ($2.00)" in text and "[pid:2] Other" in text
    assert usage.llm_calls == 0


async def test_template_branches() -> None:
    t = Synthesizer.template
    assert "escalating" in t("x", "refund", [], [], True, "value too high")
    assert t("hi", "greeting", [], [], False).startswith("Hi!")
    assert "Total $8.58" in t("cart", "cart_op", [], CART, False, "Here is your cart.")
    assert t("cart", "cart_op", [], [], False) == "Your cart is empty."
    assert t("x", "order_status", [], [], False, "Order #1 shipped.") == "Order #1 shipped."
    assert t("x", "refund", [], [], False) == "Done."
    assert "rephrase" in t("zzz", "product_search", [], [], False)
    assert t("x", "product_search", [_rp(3, "A")], [], False, "Note.").startswith(
        "Note. Top matches"
    )


async def test_llm_path_builds_context_and_usage(mocker: Any, fake_chat_factory: Any) -> None:
    chat = fake_chat_factory(["Sure! [pid:1] Heinz is great."])
    synth = Synthesizer(model="m", api_key="sk-test")
    mocker.patch.object(Synthesizer, "_build_chat", return_value=chat)
    text, usage = await synth.respond(
        message="ketchup",
        intent="product_search",
        products=[_rp(1, "Heinz")],
        cart=CART,
        escalated=True,
        note="n",
    )
    assert text.startswith("Sure!") and usage.llm_calls == 1 and usage.output_tokens == 40
    prompt = chat.calls[0][1].content
    assert (
        "<products>" in prompt and "<cart>" in prompt and "<system_note>n</system_note>" in prompt
    )
    assert "human escalation" in prompt


def test_build_chat_constructs_client() -> None:
    chat = Synthesizer(model="claude-sonnet-4-5-20250929", api_key="sk")._build_chat()
    assert chat.temperature == 0.4
