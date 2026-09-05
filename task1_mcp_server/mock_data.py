"""Deterministic mock customer/refund behavior. No real database, no external calls."""

from typing import Any


def get_customer_record(customer_id: str) -> dict[str, Any]:
    suffix = customer_id.split("-", 1)[1]
    return {
        "customer_id": customer_id,
        "name": f"Customer {suffix}",
        "status": "active",
        "email": f"{customer_id.lower()}@example.com",
    }


def trigger_refund(customer_id: str, amount: float, reason: str) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "amount": amount,
        "reason": reason,
        "status": "approved",
        "transaction_id": f"REFUND-{customer_id}-{int(round(amount * 100))}",
    }
