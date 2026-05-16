"""Refund policy escalation rules."""
from __future__ import annotations

from src.agents.refund_handler import evaluate_refund


def test_low_value_non_sensitive_auto_processes() -> None:
    d = evaluate_refund(total_usd=25.0, department="pantry", threshold_usd=100.0)
    assert d.can_auto_process is True
    assert d.requires_escalation is False


def test_above_threshold_escalates() -> None:
    d = evaluate_refund(total_usd=200.0, department="pantry", threshold_usd=100.0)
    assert d.requires_escalation is True
    assert "exceeds" in d.reason


def test_sensitive_department_always_escalates() -> None:
    d = evaluate_refund(total_usd=10.0, department="alcohol", threshold_usd=100.0)
    assert d.requires_escalation is True

    d2 = evaluate_refund(total_usd=10.0, department="babies", threshold_usd=100.0)
    assert d2.requires_escalation is True
