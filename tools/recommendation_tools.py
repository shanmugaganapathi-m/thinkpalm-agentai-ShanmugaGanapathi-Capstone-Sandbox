"""Budget recommendation tools — powered by Claude."""

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_prompt(
    summary: Dict[str, Any],
    anomalies: List[Dict],
    history: Optional[List[Dict]],
) -> str:
    month = summary.get("month", "")
    year  = summary.get("year", "")

    cats_lines = "\n".join(
        f"  - {cat}: ₹{amt:,.0f}"
        for cat, amt in sorted(
            summary.get("categories", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if amt > 0
    )

    anomaly_lines = ""
    if anomalies:
        anomaly_lines = "**Anomalies detected:**\n" + "\n".join(
            f"  - {a.get('merchant','?')} ₹{a.get('amount',0):,.0f} "
            f"({a.get('reason','').replace('_',' ')}) on {a.get('date','')}"
            for a in anomalies
        )
    else:
        anomaly_lines = "**Anomalies detected:** None"

    history_lines = ""
    if history:
        history_lines = "**Recent monthly spend (for context):**\n" + "\n".join(
            f"  - {h.get('month','')} {h.get('year','')}: "
            f"₹{h.get('total_spend',0):,.0f} spend, "
            f"{h.get('savings_rate',0):.1f}% savings rate"
            for h in history[-3:]
        )

    return f"""You are a personal finance advisor for an Indian household.
Analyze the spending data below and give exactly 4 specific, actionable recommendations.

**Statement: {month} {year}**
- Total Income:  ₹{summary.get('total_income', 0):,.0f}
- Total Spend:   ₹{summary.get('total_spend', 0):,.0f}
- Net Savings:   ₹{summary.get('net_savings', 0):,.0f}
- Savings Rate:  {summary.get('savings_rate', 0):.1f}%

**Spending by category:**
{cats_lines}

{anomaly_lines}

{history_lines}

Rules for your response:
- Give exactly 4 recommendations, numbered 1–4.
- Each must reference a real amount from the data above.
- Be specific and actionable (name the category, suggest a concrete step).
- Mention estimated savings where possible.
- Use Indian financial context (SIP, UPI cashback, RD, etc.).
- Keep each recommendation to 1–2 sentences.
- Return ONLY the numbered list — no intro, no summary.
"""


def _parse_response(text: str) -> List[str]:
    """Extract numbered list items from Claude's response."""
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering: "1.", "1)", "**1.**", "1. **" etc.
        cleaned = re.sub(r"^\*{0,2}\d+[.)]\*{0,2}\s*\*{0,2}", "", line).strip(" *")
        if len(cleaned) > 30:
            recs.append(cleaned)
    return recs[:5]


def _fallback_recommendations(
    summary: Dict[str, Any], anomalies: List[Dict]
) -> List[str]:
    """Rule-based fallback used when no API key is available."""
    tips: List[str] = []
    cats         = summary.get("categories", {})
    savings_rate = summary.get("savings_rate", 0.0)

    if (ent := cats.get("Entertainment", 0)) > 0:
        tips.append(
            f"Review entertainment subscriptions (₹{ent:,.0f}/month) — "
            f"cancelling unused ones could save ~₹{round(ent * 0.4):,.0f}."
        )
    if (food := cats.get("Food", 0)) > 0:
        tips.append(
            f"Food delivery (₹{food:,.0f}/month) can drop ~30% "
            f"(~₹{round(food * 0.3):,.0f}) by cooking at home 3–4 days a week."
        )
    if (tr := cats.get("Transport", 0)) > 0:
        tips.append(
            f"Transport costs ₹{tr:,.0f}/month — carpooling or metro "
            f"2–3 days/week could cut ~₹{round(tr * 0.2):,.0f}."
        )
    if (sh := cats.get("Shopping", 0)) > 1_000:
        tips.append(
            f"Shopping totalled ₹{sh:,.0f} — a 48-hour wishlist rule "
            f"before buying could save ~₹{round(sh * 0.25):,.0f}/month."
        )
    if savings_rate < 20:
        tips.append(
            f"Savings rate is {savings_rate:.1f}% — automating a SIP at "
            "salary credit can push this above the 20% target."
        )

    spikes = [a for a in anomalies if "spike" in a.get("reason", "")]
    if spikes:
        tips.append(
            f"Unusual spike in {spikes[0]['category']} — review transactions "
            "to separate one-off costs from new recurring charges."
        )

    return tips[:5]


def generate_recommendations(
    summary: Dict[str, Any],
    anomalies: List[Dict],
    api_key: str = "",
    history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate 4–5 personalised budget recommendations.

    Returns a dict with keys:
      - ``items``      : List[str] of recommendation strings
      - ``used_claude``: bool — True when Claude API was called successfully
      - ``error``      : str | None — error message if Claude call failed
    """
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    if not resolved_key:
        logger.warning("No API key — using rule-based recommendations.")
        return {
            "items": _fallback_recommendations(summary, anomalies),
            "used_claude": False,
            "error": None,
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=resolved_key)
        prompt = _build_prompt(summary, anomalies, history)

        logger.info("Calling Claude for recommendations…")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        raw  = response.content[0].text
        recs = _parse_response(raw)
        logger.info("Claude returned %d recommendations.", len(recs))

        if not recs:
            logger.warning("Claude response unparseable — falling back.")
            return {
                "items": _fallback_recommendations(summary, anomalies),
                "used_claude": False,
                "error": "Could not parse Claude response.",
            }

        return {"items": recs, "used_claude": True, "error": None}

    except Exception as exc:
        logger.error("Claude call failed (%s) — using fallback.", exc)
        return {
            "items": _fallback_recommendations(summary, anomalies),
            "used_claude": False,
            "error": str(exc),
        }
