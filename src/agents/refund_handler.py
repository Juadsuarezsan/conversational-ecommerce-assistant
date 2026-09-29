"""Refund policy with explicit escalation rules (documented in ``docs/decisions.md``).

A refund is auto-approved only when **both** hold:

1. the refunded value is at or below ``ESCALATION_REFUND_THRESHOLD_USD`` (default $100), and
2. none of the departments involved is in :data:`SENSITIVE_DEPARTMENTS`.

Otherwise the conversation is escalated to a human.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

#: Departments where refunds always need a human (regulated goods / safety).
SENSITIVE_DEPARTMENTS: frozenset[str] = frozenset({"alcohol", "tobacco", "pharmacy", "babies"})

_AMOUNT_RE = re.compile(r"\$\s*(\d+(?:[.,]\d{1,2})?)|(\d+(?:[.,]\d{1,2})?)\s*(?:usd|dollars|bucks)")


@dataclass(frozen=True)
class RefundDecision:
    """Outcome of the policy check."""

    can_auto_process: bool
    requires_escalation: bool
    reason: str
    amount_usd: float


def extract_amount(message: str) -> float | None:
    """Return the first dollar amount mentioned ("$200", "45 dollars"), if any."""
    m = _AMOUNT_RE.search(message.lower())
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    return float(raw.replace(",", "."))


def evaluate_refund(
    *,
    total_usd: float,
    department: str | None = None,
    departments: Iterable[str] = (),
    threshold_usd: float = 100.0,
) -> RefundDecision:
    """Apply the two-rule policy.

    Args:
        total_usd: Value being refunded.
        department: Single department involved (kept for backwards compatibility).
        departments: All departments involved.
        threshold_usd: Auto-refund ceiling.
    """
    if total_usd > threshold_usd:
        return RefundDecision(
            can_auto_process=False,
            requires_escalation=True,
            reason=f"value ${total_usd:.2f} exceeds auto-refund threshold ${threshold_usd:.2f}",
            amount_usd=total_usd,
        )
    involved = {d.lower() for d in departments if d}
    if department:
        involved.add(department.lower())
    sensitive = sorted(involved & SENSITIVE_DEPARTMENTS)
    if sensitive:
        return RefundDecision(
            can_auto_process=False,
            requires_escalation=True,
            reason=f"sensitive category: {', '.join(sensitive)}",
            amount_usd=total_usd,
        )
    return RefundDecision(
        can_auto_process=True,
        requires_escalation=False,
        reason="under threshold, non-sensitive",
        amount_usd=total_usd,
    )
