"""Anomaly detection tools for spending pattern analysis."""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

_LARGE_TXN_THRESHOLD = 10_000.0   # ₹ — flag any single expense above this
_SPIKE_MULTIPLIER = 1.5            # 150 % of 3-month average
_DUPLICATE_WINDOW_DAYS = 3         # days within which duplicates are flagged


def _parse_date(date_str: str) -> datetime:
    """Parse DD-Mon-YYYY (and a few fallback formats) to a datetime object."""
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {date_str!r}")


def detect_anomalies(
    transactions: List[Dict], history: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Detect three classes of anomaly in the current month's transactions.

    1. **Category spike** – a spending category exceeds 150 % of its
       3-month historical average (requires at least one history entry).
    2. **Large transaction** – any individual expense greater than ₹10,000.
    3. **Duplicate transaction** – same merchant and amount appear more than
       once within a 3-day window.

    Args:
        transactions: Normalized transactions for the current month
            (output of ``normalize_transactions``).
        history: List of up to 3 previous monthly summary dicts
            (output of ``calculate_monthly_summary``).  May be empty.

    Returns:
        List of anomaly dicts, each containing:
        ``date``, ``merchant``, ``amount``, ``category``, ``reason``.
        Returns an empty list when no anomalies are found.

    Example:
        >>> anomalies = detect_anomalies(transactions, history)
        >>> anomalies[0]['reason']
        'large_transaction_>10k'
    """
    anomalies: List[Dict[str, Any]] = []

    # ── 1. Category spike detection ──────────────────────────────────────────
    if history:
        recent = history[-3:]

        # Build per-category averages from history
        all_cats: set = set()
        for h in recent:
            all_cats.update(h.get("categories", {}).keys())

        cat_avgs: Dict[str, float] = {}
        for cat in all_cats:
            vals = [h.get("categories", {}).get(cat, 0.0) for h in recent]
            cat_avgs[cat] = sum(vals) / len(vals)

        # Compute current month totals per category (expenses only)
        current_totals: Dict[str, float] = {}
        for txn in transactions:
            if txn.get("type") != "expense":
                continue
            cat = txn.get("category", "Other")
            current_totals[cat] = current_totals.get(cat, 0.0) + txn["amount"]

        for cat, total in current_totals.items():
            avg = cat_avgs.get(cat, 0.0)
            if avg > 0 and total > _SPIKE_MULTIPLIER * avg:
                # Flag the largest single transaction in this spiking category
                cat_txns = [
                    t for t in transactions
                    if t.get("category") == cat and t.get("type") == "expense"
                ]
                if cat_txns:
                    worst = max(cat_txns, key=lambda t: t["amount"])
                    anomalies.append(
                        {
                            "date": worst["date"],
                            "merchant": worst["merchant"],
                            "amount": worst["amount"],
                            "category": cat,
                            "reason": "category_spike_>150%",
                        }
                    )
                    logger.debug(
                        f"Category spike: {cat} total={total:.0f} vs avg={avg:.0f}"
                    )

    # ── 2. Large individual transactions ─────────────────────────────────────
    for txn in transactions:
        if txn.get("type") == "expense" and txn.get("amount", 0) > _LARGE_TXN_THRESHOLD:
            anomalies.append(
                {
                    "date": txn["date"],
                    "merchant": txn["merchant"],
                    "amount": txn["amount"],
                    "category": txn.get("category", "Other"),
                    "reason": "large_transaction_>10k",
                }
            )
            logger.debug(f"Large transaction: {txn['merchant']} ₹{txn['amount']:.0f}")

    # ── 3. Duplicate detection ────────────────────────────────────────────────
    flagged_pairs: set[Tuple[int, int]] = set()

    for i, txn_a in enumerate(transactions):
        for j, txn_b in enumerate(transactions):
            if j <= i:
                continue
            pair = (i, j)
            if pair in flagged_pairs:
                continue
            if txn_a.get("merchant") != txn_b.get("merchant"):
                continue
            if txn_a.get("amount") != txn_b.get("amount"):
                continue
            try:
                d1 = _parse_date(txn_a["date"])
                d2 = _parse_date(txn_b["date"])
                if abs((d1 - d2).days) <= _DUPLICATE_WINDOW_DAYS:
                    flagged_pairs.add(pair)
                    anomalies.append(
                        {
                            "date": txn_a["date"],
                            "merchant": txn_a["merchant"],
                            "amount": txn_a["amount"],
                            "category": txn_a.get("category", "Other"),
                            "reason": "duplicate_transaction",
                        }
                    )
                    logger.debug(
                        f"Duplicate: {txn_a['merchant']} ₹{txn_a['amount']} "
                        f"on {txn_a['date']} and {txn_b['date']}"
                    )
            except ValueError:
                logger.debug(f"Could not parse dates for duplicate check: {txn_a}, {txn_b}")

    logger.debug(f"detect_anomalies: found {len(anomalies)} anomalies")
    return anomalies
