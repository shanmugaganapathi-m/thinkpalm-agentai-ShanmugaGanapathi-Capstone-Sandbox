"""Unit tests for all 11 tools (Phase 2)."""

import pytest
from pathlib import Path

from tools.file_tools import read_pdf, validate_pdf
from tools.parser_tools import normalize_transactions, categorize_transaction
from tools.analysis_tools import calculate_monthly_summary, compare_with_history
from tools.anomaly_tools import detect_anomalies
from tools.recommendation_tools import generate_recommendations
from memory.memory_manager import load_memory, save_memory, clear_memory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_RAW_STATEMENT = """\
Bank Statement March 2026
Date        Description                        Debit       Credit      Balance
02-Mar-2026 Salary Credit - Mirae Asset                   132000.00   132000.00
03-Mar-2026 SWIGGY FOOD ORDER                 450.00                  131550.00
05-Mar-2026 HDFC HOME LOAN EMI                28450.00                103100.00
10-Mar-2026 NETFLIX SUBSCRIPTION              649.00                  102451.00
12-Mar-2026 AMAZON SHOPPING                   1299.00                 101152.00
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 & 2 – read_pdf / validate_pdf
# ─────────────────────────────────────────────────────────────────────────────

class TestFileTool:

    def test_read_pdf_returns_text(self, sample_pdf):
        """read_pdf() extracts non-empty text from a valid PDF."""
        text = read_pdf(str(sample_pdf))
        assert text is not None
        assert len(text) > 0

    def test_read_pdf_contains_date(self, sample_pdf):
        """read_pdf() text includes the date tokens we inserted."""
        text = read_pdf(str(sample_pdf))
        assert "Mar" in text

    def test_read_pdf_missing_file_returns_none(self):
        """read_pdf() returns None for a path that doesn't exist."""
        result = read_pdf("/nonexistent/path/statement.pdf")
        assert result is None

    def test_read_pdf_wrong_extension_returns_none(self, tmp_path):
        """read_pdf() returns None for a non-.pdf file."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not a pdf")
        assert read_pdf(str(txt_file)) is None

    def test_validate_pdf_valid_file(self, sample_pdf):
        """validate_pdf() returns True for a real PDF."""
        assert validate_pdf(str(sample_pdf)) is True

    def test_validate_pdf_missing_file(self):
        """validate_pdf() returns False when file is absent."""
        assert validate_pdf("/no/such/file.pdf") is False

    def test_validate_pdf_wrong_extension(self, tmp_path):
        """validate_pdf() returns False for a .txt file."""
        f = tmp_path / "report.txt"
        f.write_text("text content")
        assert validate_pdf(str(f)) is False

    def test_validate_pdf_empty_string(self):
        """validate_pdf() returns False for an empty path string."""
        assert validate_pdf("") is False


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 – normalize_transactions
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeTransactions:

    def test_returns_list(self):
        result = normalize_transactions(_RAW_STATEMENT)
        assert isinstance(result, list)

    def test_parses_correct_count(self):
        """Five data rows in the sample text should produce 5 transactions."""
        result = normalize_transactions(_RAW_STATEMENT)
        assert len(result) == 5

    def test_income_detected(self):
        """Salary credit row is flagged as income."""
        result = normalize_transactions(_RAW_STATEMENT)
        income = [t for t in result if t["type"] == "income"]
        assert len(income) >= 1

    def test_expense_detected(self):
        """Debit rows are flagged as expense."""
        result = normalize_transactions(_RAW_STATEMENT)
        expenses = [t for t in result if t["type"] == "expense"]
        assert len(expenses) >= 1

    def test_transaction_structure(self):
        """Every transaction dict has the required keys."""
        required = {"date", "merchant", "description", "amount", "type", "category"}
        for txn in normalize_transactions(_RAW_STATEMENT):
            assert required.issubset(txn.keys())

    def test_amount_is_float(self):
        for txn in normalize_transactions(_RAW_STATEMENT):
            assert isinstance(txn["amount"], float)

    def test_empty_text_returns_empty_list(self):
        assert normalize_transactions("") == []

    def test_text_with_no_dates_returns_empty_list(self):
        assert normalize_transactions("No dates here, just plain text.") == []

    def test_date_format_preserved(self):
        """Dates should be in DD-Mon-YYYY format."""
        import re
        date_re = re.compile(r"\d{2}-[A-Za-z]{3}-\d{4}")
        for txn in normalize_transactions(_RAW_STATEMENT):
            assert date_re.match(txn["date"]), f"Bad date: {txn['date']}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 – categorize_transaction
# ─────────────────────────────────────────────────────────────────────────────

class TestCategorizeTransaction:

    @pytest.mark.parametrize("desc,expected", [
        ("SWIGGY FOOD ORDER", "Food"),
        ("ZOMATO DELIVERY", "Food"),
        ("BIGBASKET ORDER", "Groceries"),
        ("RELIANCE FRESH GROCERIES", "Groceries"),
        ("HDFC HOME LOAN EMI", "EMI/Loans"),
        ("LIC PREMIUM PAYMENT", "Insurance"),
        ("BESCOM ELECTRICITY BILL", "Utilities"),
        ("AIRTEL RECHARGE", "Utilities"),
        ("NETFLIX SUBSCRIPTION", "Entertainment"),
        ("BOOKMYSHOW MOVIE", "Entertainment"),
        ("AMAZON SHOPPING", "Shopping"),
        ("FLIPKART ORDER", "Shopping"),
        ("OLA RIDE BOOKING", "Transport"),
        ("PETROL PUMP FUEL", "Transport"),
        ("APOLLO PHARMACY", "Health"),
        ("PHARMEASY ORDER", "Health"),
        ("SIP MUTUAL FUND", "Savings/Investment"),
        ("SCHOOL FEE PAYMENT", "Education"),
        ("UDEMY COURSE FEE", "Education"),
        ("RANDOM XYZ PAYMENT", "Other"),
    ])
    def test_category_mapping(self, desc, expected):
        assert categorize_transaction(desc) == expected

    def test_case_insensitive(self):
        assert categorize_transaction("swiggy food order") == "Food"
        assert categorize_transaction("SWIGGY FOOD ORDER") == "Food"
        assert categorize_transaction("Swiggy Food Order") == "Food"

    def test_unknown_returns_other(self):
        assert categorize_transaction("COMPLETELY UNKNOWN VENDOR 12345") == "Other"

    def test_empty_string_returns_other(self):
        assert categorize_transaction("") == "Other"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 – calculate_monthly_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateMonthlySummary:

    def test_month_and_year(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        assert summary["month"] == "March"
        assert summary["year"] == 2026

    def test_total_income(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        assert summary["total_income"] == 132000.0

    def test_total_spend(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        # 450 + 28450
        assert summary["total_spend"] == pytest.approx(28900.0)

    def test_net_savings(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        assert summary["net_savings"] == pytest.approx(103100.0)

    def test_savings_rate_positive(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        assert summary["savings_rate"] > 0

    def test_all_12_categories_present(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        expected = {
            "Food", "Groceries", "EMI/Loans", "Insurance", "Utilities",
            "Entertainment", "Shopping", "Transport", "Health",
            "Savings/Investment", "Education", "Other",
        }
        assert expected == set(summary["categories"].keys())

    def test_category_amounts_non_negative(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        for cat, amt in summary["categories"].items():
            assert amt >= 0, f"{cat} has negative amount"

    def test_transaction_count(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        assert summary["transaction_count"] == len(sample_transactions)

    def test_empty_transactions(self):
        summary = calculate_monthly_summary([])
        assert summary["total_income"] == 0.0
        assert summary["total_spend"] == 0.0
        assert summary["transaction_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 – compare_with_history
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareWithHistory:

    def test_no_history_returns_first_upload(self, sample_summary):
        result = compare_with_history(sample_summary, [])
        assert result == {"status": "first_upload"}

    def test_has_history_status(self, sample_summary):
        history = [{"categories": {"Food": 1000.0}}]
        result = compare_with_history(sample_summary, history)
        assert result["status"] == "has_history"

    def test_deltas_key_present(self, sample_summary):
        history = [{"categories": {"Food": 1000.0}}]
        result = compare_with_history(sample_summary, history)
        assert "deltas" in result

    def test_spike_category_included(self, sample_summary):
        # Food in sample_summary = 2070; history avg = 1000 → +107 % > 10 %
        history = [{"categories": {"Food": 1000.0}}]
        result = compare_with_history(sample_summary, history)
        assert "Food" in result["deltas"]
        assert result["deltas"]["Food"]["change_percent"] > 10

    def test_stable_category_excluded(self, sample_summary):
        # EMI/Loans in sample_summary = 28450; history same → 0 % change
        history = [{"categories": {"EMI/Loans": 28450.0}}]
        result = compare_with_history(sample_summary, history)
        assert "EMI/Loans" not in result["deltas"]

    def test_prior_avg_in_delta(self, sample_summary):
        history = [{"categories": {"Food": 1000.0}}]
        result = compare_with_history(sample_summary, history)
        assert result["deltas"]["Food"]["prior_avg"] == pytest.approx(1000.0)

    def test_uses_at_most_3_months(self, sample_summary):
        history = [
            {"categories": {"Food": 500.0}},
            {"categories": {"Food": 500.0}},
            {"categories": {"Food": 500.0}},
            {"categories": {"Food": 500.0}},  # 4th entry – should be ignored
        ]
        result = compare_with_history(sample_summary, history)
        # avg of last 3 months = 500; current = 2070 → spike
        assert "Food" in result["deltas"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7 – detect_anomalies
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectAnomalies:

    def test_returns_list(self, sample_transactions):
        result = detect_anomalies(sample_transactions, [])
        assert isinstance(result, list)

    def test_large_transaction_flagged(self, sample_transactions):
        # HDFC HOME LOAN EMI = 28450 > 10000
        result = detect_anomalies(sample_transactions, [])
        large = [a for a in result if a["reason"] == "large_transaction_>10k"]
        assert len(large) >= 1

    def test_large_transaction_amount(self, sample_transactions):
        result = detect_anomalies(sample_transactions, [])
        large = [a for a in result if a["reason"] == "large_transaction_>10k"]
        assert all(a["amount"] > 10_000 for a in large)

    def test_anomaly_has_required_fields(self, sample_transactions):
        result = detect_anomalies(sample_transactions, [])
        required = {"date", "merchant", "amount", "category", "reason"}
        for anomaly in result:
            assert required.issubset(anomaly.keys())

    def test_no_anomalies_for_small_transactions(self):
        small_txns = [
            {"date": "01-Mar-2026", "merchant": "Swiggy", "description": "Food",
             "amount": 200.0, "type": "expense", "category": "Food"},
            {"date": "02-Mar-2026", "merchant": "Jio", "description": "Recharge",
             "amount": 299.0, "type": "expense", "category": "Utilities"},
        ]
        result = detect_anomalies(small_txns, [])
        large = [a for a in result if a["reason"] == "large_transaction_>10k"]
        assert large == []

    def test_category_spike_flagged(self, sample_summary):
        current_txns = [
            {"date": "01-Mar-2026", "merchant": "Swiggy", "description": "Food",
             "amount": 6000.0, "type": "expense", "category": "Food"},
        ]
        history = [{"categories": {"Food": 1000.0}}]  # avg 1000, current 6000 = 600%
        result = detect_anomalies(current_txns, history)
        spikes = [a for a in result if a["reason"] == "category_spike_>150%"]
        assert len(spikes) >= 1

    def test_no_spike_without_history(self):
        txns = [
            {"date": "01-Mar-2026", "merchant": "Swiggy", "description": "Food",
             "amount": 9000.0, "type": "expense", "category": "Food"},
        ]
        result = detect_anomalies(txns, [])
        spikes = [a for a in result if a["reason"] == "category_spike_>150%"]
        assert spikes == []

    def test_duplicate_transaction_flagged(self):
        txns = [
            {"date": "10-Mar-2026", "merchant": "Netflix", "description": "Netflix",
             "amount": 649.0, "type": "expense", "category": "Entertainment"},
            {"date": "11-Mar-2026", "merchant": "Netflix", "description": "Netflix",
             "amount": 649.0, "type": "expense", "category": "Entertainment"},
        ]
        result = detect_anomalies(txns, [])
        dups = [a for a in result if a["reason"] == "duplicate_transaction"]
        assert len(dups) >= 1

    def test_no_false_duplicate_beyond_window(self):
        txns = [
            {"date": "01-Mar-2026", "merchant": "Netflix", "description": "Netflix",
             "amount": 649.0, "type": "expense", "category": "Entertainment"},
            {"date": "10-Mar-2026", "merchant": "Netflix", "description": "Netflix",
             "amount": 649.0, "type": "expense", "category": "Entertainment"},
        ]
        result = detect_anomalies(txns, [])
        dups = [a for a in result if a["reason"] == "duplicate_transaction"]
        assert dups == []


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8 – generate_recommendations
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateRecommendations:

    def test_returns_list(self, sample_summary):
        recs = generate_recommendations(sample_summary, [])
        assert isinstance(recs, list)

    def test_count_between_3_and_5(self, sample_summary):
        recs = generate_recommendations(sample_summary, [])
        assert 3 <= len(recs) <= 5

    def test_each_item_is_string(self, sample_summary):
        for rec in generate_recommendations(sample_summary, []):
            assert isinstance(rec, str) and len(rec) > 0

    def test_rupee_symbol_in_recommendations(self, sample_summary):
        recs = generate_recommendations(sample_summary, [])
        assert any("₹" in r for r in recs)

    def test_anomaly_tip_included_when_spike_present(self, sample_summary):
        anomalies = [
            {"date": "10-Mar-2026", "merchant": "Amazon", "amount": 9999.0,
             "category": "Shopping", "reason": "category_spike_>150%"}
        ]
        recs = generate_recommendations(sample_summary, anomalies)
        assert any("spike" in r.lower() or "Shopping" in r for r in recs)

    def test_low_savings_rate_tip(self):
        low_savings_summary = {
            "month": "March", "year": 2026,
            "total_income": 50000.0, "total_spend": 45000.0,
            "net_savings": 5000.0, "savings_rate": 10.0,
            "categories": {
                "Food": 5000.0, "Groceries": 3000.0, "EMI/Loans": 0.0,
                "Insurance": 0.0, "Utilities": 1000.0, "Entertainment": 500.0,
                "Shopping": 2000.0, "Transport": 1500.0, "Health": 0.0,
                "Savings/Investment": 0.0, "Education": 0.0, "Other": 0.0,
            },
            "transaction_count": 15,
        }
        recs = generate_recommendations(low_savings_summary, [])
        assert any("savings" in r.lower() or "sip" in r.lower() for r in recs)


# ─────────────────────────────────────────────────────────────────────────────
# Tools 9-11 – load_memory / save_memory / clear_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManager:

    def test_load_memory_no_file_returns_empty(self, temp_memory_file):
        result = load_memory(temp_memory_file)
        assert result == {"sessions": []}

    def test_save_memory_returns_true(self, temp_memory_file, sample_summary):
        assert save_memory(sample_summary, temp_memory_file) is True

    def test_save_and_load_round_trip(self, temp_memory_file, sample_summary):
        save_memory(sample_summary, temp_memory_file)
        loaded = load_memory(temp_memory_file)
        assert len(loaded["sessions"]) == 1
        assert loaded["sessions"][0]["month"] == "March"

    def test_save_appends_multiple_sessions(self, temp_memory_file, sample_summary):
        save_memory(sample_summary, temp_memory_file)
        save_memory(sample_summary, temp_memory_file)
        loaded = load_memory(temp_memory_file)
        assert len(loaded["sessions"]) == 2

    def test_save_creates_backup(self, temp_memory_file, sample_summary):
        save_memory(sample_summary, temp_memory_file)
        save_memory(sample_summary, temp_memory_file)  # second save creates backup
        backup = Path(str(temp_memory_file) + ".backup")
        assert backup.exists()

    def test_clear_memory_returns_true(self, temp_memory_file, sample_summary):
        save_memory(sample_summary, temp_memory_file)
        assert clear_memory(temp_memory_file) is True

    def test_clear_memory_empties_sessions(self, temp_memory_file, sample_summary):
        save_memory(sample_summary, temp_memory_file)
        clear_memory(temp_memory_file)
        loaded = load_memory(temp_memory_file)
        assert loaded["sessions"] == []

    def test_clear_memory_on_nonexistent_file(self, temp_memory_file):
        """clear_memory creates the file even if it didn't exist."""
        assert clear_memory(temp_memory_file) is True
        assert temp_memory_file.exists()

    def test_load_memory_invalid_json(self, temp_memory_file):
        """Corrupt JSON returns empty store rather than raising."""
        temp_memory_file.write_text("not json {{{{")
        result = load_memory(temp_memory_file)
        assert result == {"sessions": []}
