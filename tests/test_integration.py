"""
Integration tests for the full PDF → Parser → Memory → Insights pipeline (Phase 6).

Covers multi-component workflows that cross module boundaries.
"""

import json
from pathlib import Path

import pytest

from agents.parser_agent import ParserAgent
from agents.insights_agent import InsightsAgent
from memory.memory_manager import load_memory, save_memory, clear_memory
from tools.analysis_tools import calculate_monthly_summary, compare_with_history
from tools.anomaly_tools import detect_anomalies
from tools.recommendation_tools import generate_recommendations
from tools.parser_tools import normalize_transactions, categorize_transaction


# ── Full pipeline ──────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_pdf_to_insights_complete_pipeline(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        parse_result = ParserAgent().run(str(sample_pdf))
        assert parse_result["status"] == "success"

        insights_result = InsightsAgent().run(parse_result["transactions"])
        assert insights_result["status"] == "success"
        assert insights_result["summary"]["total_income"] > 0

    def test_pipeline_month_propagates(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        parse_result = ParserAgent().run(str(sample_pdf))
        insights_result = InsightsAgent().run(parse_result["transactions"])
        assert insights_result["summary"]["month"] == parse_result["month"]

    def test_pipeline_memory_has_one_session_after_one_parse(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        ParserAgent().run(str(sample_pdf))
        assert len(load_memory(mem_file)["sessions"]) == 1

    def test_pipeline_second_parse_insights_uses_memory(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        agent = ParserAgent()
        first = agent.run(str(sample_pdf))
        assert first["status"] == "success"

        second = agent.run(str(sample_pdf))
        assert second["status"] == "success"

        insights = InsightsAgent().run(second["transactions"])
        assert insights["used_memory"] is True

    def test_pipeline_error_pdf_does_not_corrupt_memory(self, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        result = ParserAgent().run("/does/not/exist.pdf")
        assert result["status"] == "error"
        assert load_memory(mem_file) == {"sessions": []}


# ── Tools pipeline ─────────────────────────────────────────────────────────────

class TestToolsPipeline:

    def test_normalize_then_summarize(self):
        raw = (
            "02-Mar-2026 Salary Credit 132000.00 132000.00\n"
            "03-Mar-2026 SWIGGY FOOD ORDER 450.00 131550.00\n"
        )
        txns = normalize_transactions(raw)
        assert len(txns) >= 1
        summary = calculate_monthly_summary(txns)
        assert summary["total_income"] > 0 or summary["total_spend"] > 0

    def test_summary_then_compare_first_upload(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        comparison = compare_with_history(summary, [])
        assert comparison["status"] == "first_upload"

    def test_summary_then_compare_with_history(self, sample_transactions, sample_summary):
        summary = calculate_monthly_summary(sample_transactions)
        comparison = compare_with_history(summary, [sample_summary])
        assert comparison["status"] == "has_history"

    def test_anomalies_then_recommendations(self, sample_transactions, sample_summary):
        anomalies = detect_anomalies(sample_transactions, [sample_summary])
        summary = calculate_monthly_summary(sample_transactions)
        recs = generate_recommendations(summary, anomalies)
        assert isinstance(recs, list)
        assert len(recs) >= 1

    def test_full_tools_chain_no_history(self, sample_transactions):
        summary = calculate_monthly_summary(sample_transactions)
        comparison = compare_with_history(summary, [])
        anomalies = detect_anomalies(sample_transactions, [])
        recs = generate_recommendations(summary, anomalies)

        assert summary["total_income"] == 132000.0
        assert comparison["status"] == "first_upload"
        assert isinstance(anomalies, list)
        assert isinstance(recs, list)

    def test_full_tools_chain_with_history(self, sample_transactions, sample_summary):
        summary = calculate_monthly_summary(sample_transactions)
        comparison = compare_with_history(summary, [sample_summary])
        anomalies = detect_anomalies(sample_transactions, [sample_summary])
        recs = generate_recommendations(summary, anomalies)

        assert comparison["status"] == "has_history"
        assert isinstance(recs, list)


# ── Memory integration ─────────────────────────────────────────────────────────

class TestMemoryIntegration:

    def test_save_then_load_preserves_all_fields(self, sample_summary, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(sample_summary, f)
        loaded = load_memory(f)["sessions"][0]
        for key in ("month", "year", "total_income", "total_spend", "categories"):
            assert key in loaded
            assert loaded[key] == sample_summary[key]

    def test_clear_then_new_save_starts_fresh(self, sample_summary, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(sample_summary, f)
        clear_memory(f)
        save_memory({**sample_summary, "month": "April"}, f)
        sessions = load_memory(f)["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["month"] == "April"

    def test_insights_agent_uses_saved_history(self, sample_transactions, sample_summary, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        save_memory(sample_summary, mem_file)
        result = InsightsAgent().run(sample_transactions)
        assert result["used_memory"] is True
        assert result["comparison"]["status"] == "has_history"

    def test_parser_to_memory_structure(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        ParserAgent().run(str(sample_pdf))
        session = load_memory(mem_file)["sessions"][0]
        assert "total_income" in session
        assert "total_spend" in session
        assert "categories" in session
        assert isinstance(session["categories"], dict)


# ── Agent cross-interaction ───────────────────────────────────────────────────

class TestAgentInteraction:

    def test_parser_and_insights_agents_independent(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        parser = ParserAgent()
        insights = InsightsAgent()

        parse_result = parser.run(str(sample_pdf))
        insights_result = insights.run(parse_result["transactions"])

        assert parse_result["status"] == "success"
        assert insights_result["status"] == "success"

    def test_insights_result_contains_all_expected_keys(self, sample_transactions, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        result = InsightsAgent().run(sample_transactions)
        for key in ("status", "summary", "comparison", "anomalies", "recommendations", "used_memory", "history_months"):
            assert key in result

    def test_parse_result_contains_all_expected_keys(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        result = ParserAgent().run(str(sample_pdf))
        for key in ("status", "transactions", "count", "month", "year", "saved_to_memory"):
            assert key in result
