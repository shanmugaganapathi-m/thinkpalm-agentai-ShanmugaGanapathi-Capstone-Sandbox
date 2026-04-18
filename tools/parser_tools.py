"""Transaction parsing and categorization tools."""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Matches many common date formats used by Indian banks:
#   DD-Mon-YYYY  (02-Mar-2026)   DD/Mon/YYYY  DD Mon YYYY
#   DD-Mon-YY    (02-Mar-26)     DD/Mon/YY
#   DD/MM/YYYY   (02/03/2026)    DD-MM-YYYY   DD.MM.YYYY
#   DD/MM/YY     (02/03/26)      DD-MM-YY
#   YYYY-MM-DD   (2026-03-02)
_DATE_RE = re.compile(
    r"(?<!\d)("
    r"\d{1,2}[/\-. ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/\-. ]\d{4}"
    r"|\d{1,2}[/\-. ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/\-. ]\d{2}"
    r"|\d{2}[/\-.]\d{2}[/\-.]\d{4}"
    r"|\d{4}[/\-.]\d{2}[/\-.]\d{2}"
    r"|\d{2}[/\-.]\d{2}[/\-.]\d{2}"
    r")(?!\d)",
    re.IGNORECASE,
)

# Matches currency amounts with a decimal: 1,32,000.50 / 132,000.00 / 450.00
# Intentionally strict (requires decimal) so reference numbers like TXN123456 are ignored.
_AMOUNT_RE = re.compile(r"([\d,]+\.\d{1,2})")

_MONTH_NUM: Dict[str, str] = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

_MONTH_NAMES: Dict[str, str] = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
    "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
    "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec",
}


def _normalize_date(date_str: str) -> str:
    """Convert any supported date format to DD-Mon-YYYY."""
    s = date_str.strip()
    if not s:
        return date_str

    # Collapse multi-char separators (e.g. spaces used as separator) carefully.
    # Replace any separator run with a single dash.
    s_norm = re.sub(r"[/\-. ]+", "-", s)
    parts = s_norm.split("-")
    if len(parts) != 3:
        return date_str

    # If the middle part is a full month name, truncate to 3 chars so %b works.
    parts[1] = parts[1][:3].capitalize()
    s_norm = "-".join(parts)

    formats_to_try = [
        "%d-%b-%Y", "%d-%b-%y",   # DD-Mon-YYYY / DD-Mon-YY
        "%d-%m-%Y", "%d-%m-%y",   # DD-MM-YYYY  / DD-MM-YY
        "%Y-%m-%d",               # YYYY-MM-DD
    ]
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(s_norm, fmt)
            return dt.strftime("%d-%b-%Y")
        except ValueError:
            continue

    return date_str  # fallback: return as-is

# Keywords that signal a credit (income) transaction
_CREDIT_KEYWORDS = {
    "salary", "credit", "refund", "cashback", "interest",
    "dividend", "reversal", "return", "proceeds", "bonus",
}

# Category → list of lowercase keyword fragments
_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "Food": ["swiggy", "zomato", "dunzo", "blinkit", "domino", "pizza",
             "burger", "restaurant", "cafe", "dhaba", "food order"],
    "Groceries": ["bigbasket", "big basket", "reliance fresh", "dmart",
                  "grocery", "supermarket", "vegetables", "fruits", "kirana"],
    "EMI/Loans": ["home loan", "car loan", "personal loan", "housing loan",
                  "mortgage", " emi", "loan emi", "hdfc loan", "lic hfl"],
    "Insurance": ["insurance", "lic ", "health insurance", "term plan",
                  "icici pru", "bajaj allianz", "star health",
                  "insurance premium", "policy premium"],
    "Utilities": ["bescom", "airtel", "bsnl", "electricity", "water bill",
                  "gas bill", "broadband", "recharge", "jio", "vodafone",
                  "vi recharge", "bbmp", "utility"],
    "Entertainment": ["netflix", "hotstar", "prime video", "amazon prime",
                      "bookmyshow", "spotify", "youtube premium", "disney",
                      "zee5", "sonyliv", "subscription", "google play",
                      "jiocinema", "mxplayer", "voot"],
    "Shopping": ["amazon", "flipkart", "saravana", "meesho", "myntra",
                 "ajio", "nykaa", "snapdeal", "tata cliq", "online shopping"],
    "Transport": ["ola ", "rapido", "uber", "petrol", "fuel",
                  "irctc", "metro", "fastag", "toll", "cab booking",
                  "makemytrip", "redbus", "yatra", "train booking",
                  "flight booking", "bus booking"],
    "Health": ["pharmeasy", "apollo", "cult.fit", "cult fit", "medplus",
               "netmeds", "hospital", "clinic", "doctor", "pharmacy",
               "medicine", "1mg", "health check"],
    "Savings/Investment": ["sip", "ppf", "mutual fund", "zerodha", "groww",
                           "nps", "recurring deposit", "fixed deposit",
                           "elss", "equity", "investment"],
    "Education": ["school fee", "college fee", "udemy", "coursera", "byju",
                  "unacademy", "tuition", "exam fee", "course fee",
                  "education", "whitehat"],
    "Other": ["atm withdrawal", "atm cash", "google play", "miscellaneous",
              "bank charge", "processing fee"],
}


def _clean_merchant(description: str) -> str:
    """Extract a short merchant name from a full transaction description."""
    # Strip common banking prefixes
    cleaned = re.sub(
        r"^(UPI[-/]|NEFT[-/]|IMPS[-/]|RTGS[-/]|ATM[-/]|POS[-/]|ACH[-/]|ECS[-/])",
        "",
        description.strip(),
        flags=re.IGNORECASE,
    )
    # Keep first meaningful alphanumeric word as merchant
    words = [w for w in cleaned.split() if re.search(r"[A-Za-z0-9]", w)]
    return words[0].capitalize() if words else description[:20]


# Descriptions that indicate a non-transaction balance row to skip
_SKIP_DESC_PATTERNS = frozenset([
    "opening balance", "closing balance", "balance brought forward",
    "balance carried forward", "balance b/f", "balance c/f",
])

# Transaction ID prefixes to strip from descriptions
_TXN_ID_RE = re.compile(r"\b(?:TXN|REF|NREF|UTR|RRN)\w+\b", re.IGNORECASE)

# Dr / Cr markers some banks append to amounts
_DR_CR_RE = re.compile(r"\b(Dr|Cr)\b", re.IGNORECASE)


def normalize_transactions(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse raw PDF bank-statement text into structured transactions.

    Supports both single-line and multi-line-per-transaction layouts.
    Many Indian bank PDFs (IndusInd, HDFC, Axis, ICICI) extract each
    transaction as 4-6 consecutive lines:
        01 Mar 2026       ← date line
        TXN2600301002     ← transaction ID
        NACH/HDFC Loan    ← description
        28,450.00         ← debit amount
        1,04,000.00       ← closing balance

    The parser groups consecutive lines into a block starting at each
    date-bearing line, then extracts date / description / amounts from
    the combined block text.

    Args:
        raw_text: Full text extracted from a bank-statement PDF.

    Returns:
        List of dicts with keys:
        ``date``, ``merchant``, ``description``, ``amount``, ``type``,
        ``category``.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    # ── Step 1: Group lines into per-transaction blocks ────────────────
    # A new block starts every time we see a line containing a date.
    # Subsequent non-date lines belong to the same transaction (wrapped
    # narration, transaction ID, amount rows in column-extracted PDFs).
    blocks: List[List[str]] = []
    current: List[str] = []

    # Max non-date lines per block — prevents page headers from bleeding
    # into the last transaction of a page (most transactions are ≤5 lines).
    _MAX_BLOCK_LINES = 5  # date + txn-id + description + amount + balance

    for line in lines:
        if _DATE_RE.search(line):
            if current:
                blocks.append(current)
            current = [line]
        elif current and len(current) < _MAX_BLOCK_LINES:
            current.append(line)
        elif current and len(current) >= _MAX_BLOCK_LINES:
            # Block is full — flush and start fresh without a date anchor
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    # ── Step 2: Parse each block ───────────────────────────────────────
    transactions: List[Dict[str, Any]] = []

    for block in blocks:
        combined = " ".join(block)

        date_match = _DATE_RE.search(combined)
        if not date_match:
            continue

        date_str = _normalize_date(date_match.group(1))

        # Everything after the first matched date is the "payload"
        rest = combined[date_match.end():].strip()

        # Strip a second date if present (value date column)
        rest = _DATE_RE.sub("", rest).strip()

        # Strip Dr/Cr markers (capture for type hint)
        dr_cr_markers = _DR_CR_RE.findall(rest)
        rest = _DR_CR_RE.sub("", rest).strip()

        raw_amounts = _AMOUNT_RE.findall(rest)
        if not raw_amounts:
            logger.debug("Block has no decimal amounts: %r", combined[:80])
            continue

        try:
            amounts = [float(a.replace(",", "")) for a in raw_amounts]
        except ValueError:
            logger.debug("Unparseable amounts in block: %r", combined[:80])
            continue

        # Build description: strip amounts and transaction IDs
        desc = _AMOUNT_RE.sub("", rest).strip()
        desc = _TXN_ID_RE.sub("", desc).strip()
        desc = re.sub(r"\s{2,}", " ", desc).strip()

        if not desc:
            logger.debug("Empty description in block: %r", combined[:80])
            continue

        # Skip pure balance rows
        if any(p in desc.lower() for p in _SKIP_DESC_PATTERNS):
            logger.debug("Skipping balance row: %r", desc)
            continue

        desc_lower = desc.lower()
        is_credit_desc = any(kw in desc_lower for kw in _CREDIT_KEYWORDS)

        # Use Dr/Cr markers if present
        has_dr = any(m.lower() == "dr" for m in dr_cr_markers)
        has_cr = any(m.lower() == "cr" for m in dr_cr_markers)

        # ── Determine amount and type ──────────────────────────────────
        if len(amounts) >= 3:
            # Three-amount layout: debit | credit | balance
            debit_amt, credit_amt = amounts[0], amounts[1]
            if credit_amt > 0 and debit_amt == 0:
                txn_amount, txn_type = credit_amt, "income"
            elif debit_amt > 0 and credit_amt == 0:
                txn_amount, txn_type = debit_amt, "expense"
            elif credit_amt >= debit_amt:
                txn_amount, txn_type = credit_amt, "income"
            else:
                txn_amount, txn_type = debit_amt, "expense"

        elif len(amounts) == 2:
            # Two-amount layout: transaction_amount | balance
            txn_amount = amounts[0]
            if has_cr or (not has_dr and is_credit_desc):
                txn_type = "income"
            else:
                txn_type = "expense"

        else:
            # Single amount
            txn_amount = amounts[0]
            if has_cr or (not has_dr and is_credit_desc):
                txn_type = "income"
            else:
                txn_type = "expense"

        if txn_amount <= 0:
            logger.debug("Zero/negative amount — skipping: %r", desc)
            continue

        category = categorize_transaction(desc)
        merchant = _clean_merchant(desc)

        transactions.append({
            "date": date_str,
            "merchant": merchant,
            "description": desc,
            "amount": txn_amount,
            "type": txn_type,
            "category": category,
        })

    logger.debug("normalize_transactions: parsed %d transactions", len(transactions))
    return transactions


def categorize_transaction(description: str) -> str:
    """
    Classify a transaction description into one of 12 spending categories.

    Matching is case-insensitive; the first matching category wins.
    Defaults to ``"Other"`` when no keyword matches.

    Args:
        description: Full transaction description string.

    Returns:
        One of: Food, Groceries, EMI/Loans, Insurance, Utilities,
        Entertainment, Shopping, Transport, Health,
        Savings/Investment, Education, Other.

    Example:
        >>> categorize_transaction("SWIGGY FOOD ORDER")
        'Food'
        >>> categorize_transaction("NETFLIX SUBSCRIPTION")
        'Entertainment'
        >>> categorize_transaction("RANDOM XYZ PAYMENT")
        'Other'
    """
    desc_lower = description.lower()

    for category, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in desc_lower:
                logger.debug(f"Categorized '{description[:40]}' → {category}")
                return category

    logger.debug(f"Categorized '{description[:40]}' → Other (no match)")
    return "Other"
