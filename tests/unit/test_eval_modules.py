"""Eval set integrity, judge rubric, baselines, RAGAS rows, conversation checks."""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.api.schemas import CartItem, ChatResponse
from src.eval.baselines import (
    BaselineUnavailable,
    ZeroShotBaseline,
    evaluate_zero_shot,
    match_names_to_catalog,
)
from src.eval.conversation import check_expectations
from src.eval.judge import (
    RUBRIC,
    SYSTEM_PROMPT,
    JudgeUnavailable,
    LLMJudge,
    format_transcript,
    judge_conversation,
)
from src.eval.ragas_eval import RagasUnavailable, build_ragas_rows, run_ragas
from src.eval.runner import load_eval_set, relevance_map


def test_eval_set_has_spec_sizes_and_valid_ids(catalog: list[dict[str, Any]]) -> None:
    records = load_eval_set()
    assert len(records) == 100
    assert Counter(r["category"] for r in records) == {
        "lookup": 30,
        "semantic": 30,
        "comparative": 20,
        "multi_turn": 20,
    }
    ids = {p["product_id"] for p in catalog}
    assert len({r["id"] for r in records}) == 100
    for r in records:
        for rel in r["relevant"]:
            assert rel["product_id"] in ids and 1 <= rel["relevance"] <= 3
        if r["category"] == "multi_turn":
            assert r["setup_turns"] and r["expect"] and r["context"]
        else:
            assert r["relevant"], r["id"]
    assert relevance_map(records[0]) == {1: 3, 2: 2}


def _resp(**kw: Any) -> ChatResponse:
    base = {
        "session_id": "s",
        "turn": 1,
        "intent": "cart_op",
        "confidence": 1.0,
        "response": "",
        "latency_ms": 1,
    }
    return ChatResponse(**{**base, **kw})


def test_check_expectations() -> None:
    cart = [CartItem(product_id=7, product_name="A", quantity=2, unit_price_usd=1.0)]
    r = _resp(cart_snapshot=cart, response="[pid:7] ok", escalated=False)
    assert (
        check_expectations(
            {
                "intent": "cart_op",
                "cart_has": [7],
                "cart_has_any": [7, 9],
                "cart_has_rank": 2,
                "qty_rank": 2,
                "cart_size": 1,
                "response_contains": ["ok"],
                "response_mentions_any": [7],
                "escalated": False,
            },
            r,
            [6, 7],
        )
        == []
    )
    failed = check_expectations(
        {
            "intent": "refund",
            "cart_has": [1],
            "cart_has_any": [1],
            "cart_has_rank": 1,
            "cart_qty_rank": {"rank": 2, "qty": 3},
            "cart_size": 0,
            "cart_empty": True,
            "response_contains": ["zzz"],
            "response_mentions_any": [1],
            "escalated": True,
        },
        r,
        [6, 7],
    )
    assert set(failed) == {
        "intent",
        "cart_has",
        "cart_has_any",
        "cart_has_rank",
        "cart_qty_rank",
        "cart_size",
        "cart_empty",
        "response_contains",
        "response_mentions_any",
        "escalated",
    }
    assert check_expectations({"cart_has_rank": 1, "qty_rank": 3}, r, [7]) == ["qty_rank"]


def test_rubric_is_numbered_in_prompt() -> None:
    assert [n for n, _, _ in RUBRIC] == [1, 2, 3, 4, 5]
    for n, name, _ in RUBRIC:
        assert f"{n}. {name}:" in SYSTEM_PROMPT
    text = format_transcript([{"user": "hi", "assistant": "hello", "intent": "greeting"}], [(1, 2)])
    assert "[1] USER: hi" in text and "pid 1 x2" in text


async def test_judge_without_key_and_with_mock(mocker: Any, fake_chat_factory: Any) -> None:
    assert LLMJudge(api_key="sk")._build_chat().temperature == 0.0
    with pytest.raises(JudgeUnavailable):
        await LLMJudge(api_key="").judge("t")
    judge = LLMJudge(api_key="sk")
    mocker.patch.object(
        LLMJudge,
        "_build_chat",
        return_value=fake_chat_factory(
            ['{"scores": {"1": 1, "2": 1, "3": 0, "4": 1, "5": 1}, "rationale": "r"}']
        ),
    )
    v = await judge_conversation(
        {"turns": [{"user": "u", "assistant": "a", "intent": "x", "cart": [(1, 1)]}]}, judge
    )
    assert v.task_completed and v.total == 4
    mocker.patch.object(LLMJudge, "_build_chat", return_value=fake_chat_factory(["bad"] * 3))
    mocker.patch.object(LLMJudge.judge.retry, "sleep", AsyncMock())
    with pytest.raises(ValueError):
        await judge.judge("t")


async def test_zero_shot_baseline(mocker: Any, fake_chat_factory: Any, bm25_index: Any) -> None:
    assert ZeroShotBaseline(api_key="sk")._build_chat().max_retries == 0
    assert match_names_to_catalog(
        ["Heinz Tomato Ketchup 397g", "Heinz Tomato Ketchup 397g", "zzzz"], bm25_index
    ) == [1]
    with pytest.raises(BaselineUnavailable):
        await ZeroShotBaseline(api_key="").suggest("x")
    zs = ZeroShotBaseline(api_key="sk")
    mocker.patch.object(
        ZeroShotBaseline,
        "_build_chat",
        return_value=fake_chat_factory(['{"products": ["Heinz Tomato Ketchup 397g"]}']),
    )
    report = await evaluate_zero_shot(
        [{"id": "q", "query": "ketchup", "relevant": [{"product_id": 1, "relevance": 3}]}],
        bm25_index,
        zs,
    )
    assert report["n_queries"] == 1 and report["overall"]["mrr"] == 1.0
    mocker.patch.object(
        ZeroShotBaseline, "_build_chat", return_value=fake_chat_factory(["nope"] * 3)
    )
    mocker.patch.object(ZeroShotBaseline.suggest.retry, "sleep", AsyncMock())
    with pytest.raises(ValueError):
        await zs.suggest("x")


async def test_ragas_rows_and_gate(runtime: Any) -> None:
    rows = build_ragas_rows(
        [
            {
                "query": "q",
                "relevant": [{"product_id": 1, "relevance": 3}, {"product_id": 9, "relevance": 2}],
            },
            {"query": "q2", "relevant": []},
        ],
        ["a1", "a2"],
        [["c1"], []],
        {1: "Heinz"},
    )
    assert rows[0]["ground_truth"] == "Heinz" and rows[1]["ground_truth"].startswith("no exact")
    with pytest.raises(RagasUnavailable):
        await run_ragas([], runtime)
    from src.eval.ragas_eval import collect_answers

    answers, contexts = await collect_answers([{"id": "x", "query": "Heinz ketchup"}], runtime)
    assert "[pid:" in answers[0] and contexts[0]
