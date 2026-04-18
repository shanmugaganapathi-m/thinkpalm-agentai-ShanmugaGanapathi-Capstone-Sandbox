"""
Phase 4 memory integration tests.

Covers:
  - memory_manager: load / save / clear (enhanced behaviours)
  - ParserAgent: saves session to memory after parse
  - InsightsAgent: auto-loads history from memory
  - Integration: full Parser → Memory → Insights pipeline
"""

import json
from pathlib import Path

import pytest

from memory.memory_manager import clear_memory, load_memory, save_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(month: str = "March", year: int = 2026, spend: float = 50000.0) -> dict:
    return {
        "month": month,
        "year": year,
        "total_income": 132000.0,
        "total_spend": spend,
        "net_savings": 132000.0 - spend,
        "savings_rate": round((132000.0 - spend) / 132000.0 * 100, 2),
        "transaction_count": 20,
        "categories": {"Food": 2000.0, "EMI/Loans": spend - 2000.0},
        "anomaly_count": 0,
        "recommendations_count": 3,
    }


# ---------------------------------------------------------------------------
# load_memory
# ---------------------------------------------------------------------------

class TestLoadMemory:

    def test_load_returns_empty_when_file_missing(self, tmp_path):
        result = load_memory(tmp_path / "nonexistent.json")
        assert result == {"sessions": []}

    def test_load_returns_empty_on_empty_file(self, tmp_path):
        f = tmp_path / "mem.json"
        f.write_text("", encoding="utf-8")
        assert load_memory(f) == {"sessions": []}

    def test_load_returns_empty_on_corrupted_json(self, tmp_path):
        f = tmp_path / "mem.json"
        f.write_text("{bad json!!}", encoding="utf-8")
        assert load_memory(f) == {"sessions": []}

    def test_load_returns_empty_on_missing_sessions_key(self, tmp_path):
        f = tmp_path / "mem.json"
        f.write_text(json.dumps({"data": []}), encoding="utf-8")
        assert load_memory(f) == {"sessions": []}

    def test_load_returns_sessions_from_valid_file(self, tmp_path):
        session = _make_session()
        f = tmp_path / "mem.json"
        f.write_text(json.dumps({"sessions": [session]}), encoding="utf-8")
        result = load_memory(f)
        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["month"] == "March"

    def test_load_preserves_multiple_sessions(self, tmp_path):
        sessions = [_make_session("January"), _make_session("February"), _make_session("March")]
        f = tmp_path / "mem.json"
        f.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
        result = load_memory(f)
        assert len(result["sessions"]) == 3


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------

class TestSaveMemory:

    def test_save_returns_true_on_success(self, tmp_path):
        assert save_memory(_make_session(), tmp_path / "mem.json") is True

    def test_save_creates_file_if_missing(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session(), f)
        assert f.exists()

    def test_save_appends_to_existing_sessions(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session("January"), f)
        save_memory(_make_session("February"), f)
        result = load_memory(f)
        assert len(result["sessions"]) == 2

    def test_save_creates_backup_file(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session("January"), f)
        save_memory(_make_session("February"), f)  # second save triggers backup
        backup = f.with_suffix(f.suffix + ".backup")
        assert backup.exists()

    def test_save_rolling_window_keeps_12(self, tmp_path):
        f = tmp_path / "mem.json"
        months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan",  # 13th entry
        ]
        for m in months:
            save_memory(_make_session(m), f)
        result = load_memory(f)
        assert len(result["sessions"]) == 12

    def test_save_rolling_window_drops_oldest(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session("January", 2025), f)
        for i in range(12):
            save_memory(_make_session(f"Month{i}", 2026), f)
        result = load_memory(f)
        assert all(s["year"] == 2026 for s in result["sessions"])

    def test_save_returns_false_for_empty_session(self, tmp_path):
        assert save_memory({}, tmp_path / "mem.json") is False

    def test_save_returns_false_for_none_session(self, tmp_path):
        assert save_memory(None, tmp_path / "mem.json") is False  # type: ignore

    def test_save_and_load_round_trip(self, tmp_path):
        f = tmp_path / "mem.json"
        session = _make_session("March", 2026, 60000.0)
        save_memory(session, f)
        loaded = load_memory(f)
        assert loaded["sessions"][0]["month"] == "March"
        assert loaded["sessions"][0]["total_spend"] == 60000.0


# ---------------------------------------------------------------------------
# clear_memory
# ---------------------------------------------------------------------------

class TestClearMemory:

    def test_clear_returns_true(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session(), f)
        assert clear_memory(f) is True

    def test_clear_empties_sessions(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session(), f)
        clear_memory(f)
        assert load_memory(f) == {"sessions": []}

    def test_clear_creates_backup(self, tmp_path):
        f = tmp_path / "mem.json"
        save_memory(_make_session(), f)
        clear_memory(f)
        backup = f.with_suffix(f.suffix + ".backup")
        assert backup.exists()

    def test_clear_on_nonexistent_file_returns_true(self, tmp_path):
        assert clear_memory(tmp_path / "ghost.json") is True


# ---------------------------------------------------------------------------
# ParserAgent memory integration
# ---------------------------------------------------------------------------

class TestParserAgentMemory:

    def test_parser_run_includes_saved_to_memory_key(self, sample_pdf):
        from agents.parser_agent import ParserAgent
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert "saved_to_memory" in result

    def test_parser_run_saves_successfully(self, sample_pdf, tmp_path, monkeypatch):
        """Parser should save a session and return saved_to_memory=True."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.parser_agent import ParserAgent
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"
        assert result["saved_to_memory"] is True

    def test_parser_saved_session_structure(self, sample_pdf, tmp_path, monkeypatch):
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.parser_agent import ParserAgent
        ParserAgent().run(str(sample_pdf))

        sessions = load_memory(mem_file)["sessions"]
        assert len(sessions) == 1
        s = sessions[0]
        for key in ("month", "year", "total_income", "total_spend", "categories"):
            assert key in s

    def test_parser_error_returns_saved_to_memory_false(self):
        from agents.parser_agent import ParserAgent
        result = ParserAgent().run("/nonexistent/path.pdf")
        assert result["status"] == "error"
        assert result["saved_to_memory"] is False

    def test_parser_create_session_summary_has_required_fields(self, sample_transactions):
        from agents.parser_agent import ParserAgent
        agent = ParserAgent()
        summary = agent._create_session_summary(sample_transactions)
        for key in ("month", "year", "total_income", "total_spend",
                    "net_savings", "savings_rate", "categories",
                    "anomaly_count", "recommendations_count"):
            assert key in summary


# ---------------------------------------------------------------------------
# InsightsAgent memory integration
# ---------------------------------------------------------------------------

class TestInsightsAgentMemory:

    def test_insights_run_includes_used_memory_key(self, sample_transactions):
        from agents.insights_agent import InsightsAgent
        result = InsightsAgent().run(sample_transactions)
        assert "used_memory" in result
        assert "history_months" in result

    def test_insights_no_history_used_memory_false(self, sample_transactions, tmp_path, monkeypatch):
        """With empty memory file, used_memory should be False."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.insights_agent import InsightsAgent
        result = InsightsAgent().run(sample_transactions)
        assert result["status"] == "success"
        assert result["used_memory"] is False
        assert result["history_months"] == 0

    def test_insights_loads_history_from_memory(self, sample_transactions, sample_summary, tmp_path, monkeypatch):
        """After saving a session, InsightsAgent should auto-load it."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        save_memory(sample_summary, mem_file)

        from agents.insights_agent import InsightsAgent
        result = InsightsAgent().run(sample_transactions)
        assert result["status"] == "success"
        assert result["used_memory"] is True
        assert result["history_months"] == 1

    def test_insights_explicit_history_overrides_memory(self, sample_transactions, sample_summary):
        """When history is explicitly passed, memory should not be loaded."""
        from agents.insights_agent import InsightsAgent
        result = InsightsAgent().run(sample_transactions, history=[sample_summary])
        assert result["used_memory"] is True
        assert result["history_months"] == 1

    def test_insights_empty_history_list_not_used_memory(self, sample_transactions):
        from agents.insights_agent import InsightsAgent
        result = InsightsAgent().run(sample_transactions, history=[])
        assert result["used_memory"] is False
        assert result["history_months"] == 0


# ---------------------------------------------------------------------------
# Integration: Parser → Memory → Insights pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_parser_to_insights_via_memory(self, sample_pdf, tmp_path, monkeypatch):
        """Full pipeline: parse PDF → save to memory → insights loads history."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.parser_agent import ParserAgent
        from agents.insights_agent import InsightsAgent

        parse_result = ParserAgent().run(str(sample_pdf))
        assert parse_result["status"] == "success"
        assert parse_result["saved_to_memory"] is True

        insights_result = InsightsAgent().run(parse_result["transactions"])
        assert insights_result["status"] == "success"
        assert insights_result["used_memory"] is True

    def test_memory_state_after_pipeline(self, sample_pdf, tmp_path, monkeypatch):
        """Memory should contain exactly one session after one parse."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.parser_agent import ParserAgent
        ParserAgent().run(str(sample_pdf))

        sessions = load_memory(mem_file)["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["month"] == "March"
        assert sessions[0]["year"] == 2026

    def test_two_parses_accumulate_in_memory(self, sample_pdf, tmp_path, monkeypatch):
        """Parsing the same PDF twice should store two sessions."""
        mem_file = tmp_path / "mem.json"
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from agents.parser_agent import ParserAgent
        agent = ParserAgent()
        agent.run(str(sample_pdf))
        agent.run(str(sample_pdf))

        assert len(load_memory(mem_file)["sessions"]) == 2
