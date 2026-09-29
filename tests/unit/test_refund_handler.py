"""Refund policy rules and amount extraction."""

from __future__ import annotations

from src.agents.refund_handler import evaluate_refund, extract_amount


def test_low_value_non_sensitive_auto_processes() -> None:
    d = evaluate_refund(total_usd=25.0, department="pantry", threshold_usd=100.0)
    assert d.can_auto_process and not d.requires_escalation and d.amount_usd == 25.0


def test_above_threshold_escalates() -> None:
    d = evaluate_refund(total_usd=200.0, department="pantry", threshold_usd=100.0)
    assert d.requires_escalation and "exceeds" in d.reason


def test_sensitive_department_always_escalates() -> None:
    assert evaluate_refund(total_usd=10.0, department="alcohol").requires_escalation
    assert evaluate_refund(total_usd=10.0, departments=["pantry", "Babies"]).requires_escalation
    d = evaluate_refund(total_usd=10.0, departments=["pantry", "produce"])
    assert not d.requires_escalation


def test_extract_amount() -> None:
    assert extract_amount("refund my $200 order") == 200.0
    assert extract_amount("I paid 45.50 dollars") == 45.5
    assert extract_amount("return the bread") is None
