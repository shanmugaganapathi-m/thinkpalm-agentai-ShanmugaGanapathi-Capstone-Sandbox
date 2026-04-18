"""Tools for analyzing monthly spending patterns."""

import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_EXPENSE_CATEGORIES = [
    "Food", "Groceries", "EMI/Loans", "Insurance", "Utilities",
    "Entertainment", "Shopping", "Transport", "Health",
    "Savings/Investment", "Education", "Other",
]


def calculate_monthly_summary(transactions: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate a list of normalized transactions into a monthly summary.

    Income transactions (``type == "income"``) contribute only to
    ``total_income``; expense transactions contribute to ``total_spend``
    and their respective category buckets.

    Args:
        transactions: List of normalized transaction dicts produced by
            ``normalize_transactions()``.  Each dict must contain at
            minimum ``type``, ``amount``, ``category``, and ``date`` keys.

    Returns:
        Dict with keys:
        ``month`` (str), ``year`` (int), ``total_income`` (float),
        ``total_spend`` (float), ``net_savings`` (float),
        ``savings_rate`` (float, percentage), ``categories`` (dict),
        ``transaction_count`` (int).

    Example:
        >>> summary = calculate_monthly_summary(transactions)
        >>> summary['total_income']
        132000.0
        >>> summary['categories']['Food']
        450.0
    """
    total_income = 0.0
    total_spend = 0.0
    categories: Dict[str, float] = {cat: 0.0 for cat in _EXPENSE_CATEGORIES}

    for txn in transactions:
        amt = float(txn.get("amount", 0))
        if txn.get("type") == "income":
            total_income += amt
        else:
            total_spend += amt
            cat = txn.get("category", "Other")
            if cat in categories:
                categories[cat] += amt
            else:
                categories["Other"] += amt

    net_savings = total_income - total_spend
    savings_rate = round((net_savings / total_income * 100), 2) if total_income > 0 else 0.0

    month = ""
    year = 0
    if transactions:
        date_str = transactions[0].get("date", "")
        for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                month = dt.strftime("%B")
                year = dt.year
                break
            except ValueError:
                continue

    summary = {
        "month": month,
        "year": year,
        "total_income": total_income,
        "total_spend": total_spend,
        "net_savings": net_savings,
        "savings_rate": savings_rate,
        "categories": categories,
        "transaction_count": len(transactions),
    }
    logger.debug(
        f"Monthly summary: income={total_income}, spend={total_spend}, "
        f"savings_rate={savings_rate}%"
    )
    return summary


def compare_with_history(
    current_summary: Dict, history: List[Dict]
) -> Dict[str, Any]:
    """
    Compare the current month's category spend against historical averages.

    Uses up to the last three months in ``history`` to compute per-category
    averages.  Only categories that changed by more than 10 % are included
    in ``deltas``.

    Args:
        current_summary: Monthly summary dict for the current period
            (output of ``calculate_monthly_summary``).
        history: Ordered list of previous monthly summaries, oldest first.
            Each element must contain a ``categories`` dict.

    Returns:
        ``{"status": "first_upload"}`` when history is empty, otherwise
        ``{"status": "has_history", "deltas": {category: {"change_percent": float,
        "prior_avg": float}, ...}}``.

    Example:
        >>> result = compare_with_history(march_summary, [feb_summary])
        >>> result['status']
        'has_history'
        >>> result['deltas']['Food']['change_percent']
        32.5
    """
    if not history:
        logger.debug("No history found – returning first_upload status")
        return {"status": "first_upload"}

    recent = history[-3:]  # use at most 3 prior months
    current_cats = current_summary.get("categories", {})
    deltas: Dict[str, Any] = {}

    for cat, current_amt in current_cats.items():
        hist_vals = [h.get("categories", {}).get(cat, 0.0) for h in recent]
        prior_avg = sum(hist_vals) / len(hist_vals)

        if prior_avg <= 0:
            continue

        change_pct = (current_amt - prior_avg) / prior_avg * 100
        if abs(change_pct) > 10:
            deltas[cat] = {
                "change_percent": round(change_pct, 1),
                "prior_avg": round(prior_avg, 2),
            }

    logger.debug(f"compare_with_history: {len(deltas)} categories changed >10%")
    return {"status": "has_history", "deltas": deltas}
