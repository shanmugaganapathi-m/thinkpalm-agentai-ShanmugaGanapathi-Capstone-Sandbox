"""
Edge case and boundary condition tests (Phase 6).

Covers: empty data, extreme values, special characters,
exact boundary conditions, and unusual input shapes.
"""

import pytest

from tools.parser_tools import normalize_transactions, categorize_transaction, _clean_merchant
from tools.analysis_tools import calculate_monthly_summary, compare_with_history
from tools.anomaly_tools import detect_anomalies, _parse_date
from tools.recommendation_tools import generate_recommendations
from memory.memory_manager import load_memory, save_memory


# ── _clean_merchant ────────────────────────────────────────────────────────────

class TestCleanMerchant:

    def test_strips_upi_prefix(self):
        assert _clean_merchant("UPI/SWIGGY TECHNOLOGIES") == "Swiggy"

    def test_strips_neft_prefix(self):
        result = _clean_merchant("NEFT-HDFC BANK TRANSFER")
        assert "Upi" not in result and "Neft" not in result

    def test_strips_imps_prefix(self):
        result = _clean_merchant("IMPS/PAYTM PAYMENT")
        assert result != ""

    def test_strips_pos_prefix(self):
        result = _clean_merchant("POS/AMAZON RETAIL")
        assert result != ""

    def test_empty_description_returns_something(self):
        result = _clean_merchant("")
        assert isinstance(result, str)

    def test_single_word_description(self):
        assert _clean_merchant("Salary") == "Salary"

    def test_preserves_capitalization(self):
        result = _clean_merchant("zomato food order")
        assert result[0].isupper()


# ── _parse_date ────────────────────────────────────────────────────────────────

class TestParseDate:

    def test_dd_mon_yyyy_format(self):
        dt = _parse_date("02-Mar-2026")
        assert dt.day == 2
        assert dt.month == 3
        assert dt.year == 2026

    def test_dd_slash_mm_slash_yyyy_format(self):
        dt = _parse_date("02/03/2026")
        assert dt.month == 3

    def test_yyyy_mm_dd_format(self):
        dt = _parse_date("2026-03-02")
        assert dt.year == 2026

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")

    def test_case_insensitive_month(self):
        dt = _parse_date("02-mar-2026")
        assert dt.month == 3


# ── normalize_transactions edge cases ─────────────────────────────────────────

class TestNormalizeEdgeCases:

    def test_empty_string_returns_empty_list(self):
        assert normalize_transactions("") == []

    def test_no_dates_returns_empty_list(self):
        assert normalize_transactions("Hello world\nNo dates here") == []

    def test_line_with_date_but_no_amount_skipped(self):
        result = normalize_transactions("02-Mar-2026 Salary Credit no amounts here")
        assert result == []

    def test_commas_in_amounts_parsed(self):
        raw = "02-Mar-2026 Salary Credit 1,32,000.00 1,32,000.00"
        result = normalize_transactions(raw)
        assert len(result) >= 1
        assert result[0]["amount"] == pytest.approx(132000.0)

    def test_special_characters_in_description(self):
        raw = "02-Mar-2026 UPI/CAFÉ & RESTO 250.00 5000.00"
        result = normalize_transactions(raw)
        assert len(result) >= 1

    def test_multiple_transactions_parsed(self):
        raw = (
            "02-Mar-2026 Salary 132000.00 132000.00\n"
            "03-Mar-2026 SWIGGY 450.00 131550.00\n"
            "05-Mar-2026 HDFC EMI 28450.00 103100.00\n"
        )
        result = normalize_transactions(raw)
        assert len(result) == 3


# ── calculate_monthly_summary boundaries ──────────────────────────────────────

class TestSummaryBoundaries:

    def test_all_income_no_expenses(self):
        txns = [{"date": "01-Mar-2026", "amount": 50000.0, "type": "income", "category": "Income", "merchant": "x", "description": "y"}]
        summary = calculate_monthly_summary(txns)
        assert summary["total_spend"] == 0.0
        assert summary["savings_rate"] == 100.0

    def test_all_expenses_no_income(self):
        txns = [{"date": "01-Mar-2026", "amount": 5000.0, "type": "expense", "category": "Food", "merchant": "x", "description": "y"}]
        summary = calculate_monthly_summary(txns)
        assert summary["total_income"] == 0.0
        assert summary["savings_rate"] == 0.0

    def test_zero_income_savings_rate_zero(self):
        txns = [{"date": "01-Mar-2026", "amount": 0.0, "type": "income", "category": "Income", "merchant": "x", "description": "y"}]
        summary = calculate_monthly_summary(txns)
        assert summary["savings_rate"] == 0.0

    def test_single_transaction(self):
        txns = [{"date": "01-Mar-2026", "amount": 1000.0, "type": "expense", "category": "Food", "merchant": "x", "description": "y"}]
        summary = calculate_monthly_summary(txns)
        assert summary["transaction_count"] == 1

    def test_unknown_category_goes_to_other(self):
        txns = [{"date": "01-Mar-2026", "amount": 500.0, "type": "expense", "category": "UnknownCat", "merchant": "x", "description": "y"}]
        summary = calculate_monthly_summary(txns)
        assert summary["categories"]["Other"] == 500.0


# ── generate_recommendations boundary conditions ───────────────────────────────

class TestRecommendationBoundaries:

    def test_shopping_exactly_1000_no_tip(self):
        """Shopping at exactly ₹1,000 should NOT trigger the shopping tip."""
        summary = {
            "categories": {"Shopping": 1000.0, "Food": 0.0, "Transport": 0.0, "Entertainment": 0.0},
            "savings_rate": 25.0,
            "total_spend": 1000.0,
        }
        recs = generate_recommendations(summary, [])
        assert not any("wishlist" in r.lower() for r in recs)

    def test_shopping_above_1000_triggers_tip(self):
        """Shopping at ₹1,001 should trigger the shopping tip."""
        summary = {
            "categories": {"Shopping": 1001.0, "Food": 0.0, "Transport": 0.0, "Entertainment": 0.0},
            "savings_rate": 25.0,
            "total_spend": 1001.0,
        }
        recs = generate_recommendations(summary, [])
        assert any("wishlist" in r.lower() for r in recs)

    def test_savings_rate_exactly_20_no_sip_tip(self):
        """Savings rate at exactly 20% should NOT trigger the SIP tip."""
        summary = {
            "categories": {"Shopping": 0.0, "Food": 0.0, "Transport": 0.0, "Entertainment": 0.0},
            "savings_rate": 20.0,
            "total_spend": 0.0,
        }
        recs = generate_recommendations(summary, [])
        assert not any("sip" in r.lower() for r in recs)

    def test_savings_rate_below_20_triggers_sip_tip(self):
        """Savings rate below 20% should trigger the SIP tip."""
        summary = {
            "categories": {"Shopping": 0.0, "Food": 0.0, "Transport": 0.0, "Entertainment": 0.0},
            "savings_rate": 19.9,
            "total_spend": 0.0,
        }
        recs = generate_recommendations(summary, [])
        assert any("sip" in r.lower() for r in recs)

    def test_max_5_recommendations(self):
        """Should never produce more than 5 recommendations."""
        summary = {
            "categories": {
                "Shopping": 5000.0, "Food": 3000.0, "Transport": 1500.0, "Entertainment": 800.0,
            },
            "savings_rate": 10.0,
            "total_spend": 10300.0,
        }
        anomalies = [{"reason": "category_spike_>150%", "category": "Food"}]
        recs = generate_recommendations(summary, anomalies)
        assert len(recs) <= 5

    def test_empty_categories_returns_empty_or_sip_tip(self):
        """With all zero categories and low savings rate, only SIP tip (or none)."""
        summary = {
            "categories": {cat: 0.0 for cat in ["Food", "Shopping", "Transport", "Entertainment"]},
            "savings_rate": 5.0,
            "total_spend": 0.0,
        }
        recs = generate_recommendations(summary, [])
        assert isinstance(recs, list)


# ── detect_anomalies edge cases ───────────────────────────────────────────────

class TestAnomalyEdgeCases:

    def test_exactly_at_threshold_not_flagged(self):
        """Expense at exactly ₹10,000 should NOT be flagged as large."""
        txns = [{"date": "01-Mar-2026", "merchant": "Test", "amount": 10000.0, "type": "expense", "category": "Other"}]
        anomalies = detect_anomalies(txns, [])
        large = [a for a in anomalies if a["reason"] == "large_transaction_>10k"]
        assert len(large) == 0

    def test_just_above_threshold_flagged(self):
        """Expense at ₹10,001 should be flagged as large."""
        txns = [{"date": "01-Mar-2026", "merchant": "Test", "amount": 10001.0, "type": "expense", "category": "Other"}]
        anomalies = detect_anomalies(txns, [])
        large = [a for a in anomalies if a["reason"] == "large_transaction_>10k"]
        assert len(large) == 1

    def test_income_not_flagged_as_large(self):
        """Income transactions should never be flagged as large."""
        txns = [{"date": "01-Mar-2026", "merchant": "Salary", "amount": 200000.0, "type": "income", "category": "Income"}]
        anomalies = detect_anomalies(txns, [])
        large = [a for a in anomalies if a["reason"] == "large_transaction_>10k"]
        assert len(large) == 0

    def test_duplicate_outside_window_not_flagged(self):
        """Duplicate transactions outside the 3-day window should not be flagged."""
        txns = [
            {"date": "01-Mar-2026", "merchant": "Netflix", "amount": 649.0, "type": "expense", "category": "Entertainment"},
            {"date": "10-Mar-2026", "merchant": "Netflix", "amount": 649.0, "type": "expense", "category": "Entertainment"},
        ]
        anomalies = detect_anomalies(txns, [])
        dupes = [a for a in anomalies if a["reason"] == "duplicate_transaction"]
        assert len(dupes) == 0

    def test_duplicate_within_window_flagged(self):
        """Same merchant and amount within 3 days should be flagged as duplicate."""
        txns = [
            {"date": "01-Mar-2026", "merchant": "Netflix", "amount": 649.0, "type": "expense", "category": "Entertainment"},
            {"date": "02-Mar-2026", "merchant": "Netflix", "amount": 649.0, "type": "expense", "category": "Entertainment"},
        ]
        anomalies = detect_anomalies(txns, [])
        dupes = [a for a in anomalies if a["reason"] == "duplicate_transaction"]
        assert len(dupes) == 1

    def test_empty_transactions_no_anomalies(self):
        assert detect_anomalies([], []) == []

    def test_no_history_no_category_spike(self, sample_transactions):
        """Without history, category spike detection should not run."""
        anomalies = detect_anomalies(sample_transactions, [])
        spikes = [a for a in anomalies if "spike" in a["reason"]]
        assert len(spikes) == 0
