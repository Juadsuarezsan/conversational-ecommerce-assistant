"""Refund policy — explicit escalation thresholds, no implicit silent magic."""
from __future__ import annotations

from dataclasses import dataclass

SENSITIVE_DEPARTMENTS = {"alcohol", "tobacco", "pharmacy", "babies"}


@dataclass
class RefundDecision:
    can_auto_process: bool
    requires_escalation: bool
    reason: str


def evaluate_refund(*, total_usd: float, department: str | None, threshold_usd: float) -> RefundDecision:
    if total_usd > threshold_usd:
        return RefundDecision(
            can_auto_process=False, requires_escalation=True,
            reason=f"value ${total_usd:.2f} exceeds auto-refund threshold ${threshold_usd:.2f}",
        )
    if department and department.lower() in SENSITIVE_DEPARTMENTS:
        return RefundDecision(
            can_auto_process=False, requires_escalation=True,
            reason=f"sensitive category: {department}",
        )
    return RefundDecision(can_auto_process=True, requires_escalation=False, reason="under threshold, non-sensitive")
