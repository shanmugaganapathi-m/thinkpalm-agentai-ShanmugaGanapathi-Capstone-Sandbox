"""
End-to-end user scenario tests (Phase 6).

Simulates realistic user workflows through the application from upload to insights.
"""

import pytest

from agents.parser_agent import ParserAgent
from agents.insights_agent import InsightsAgent
from memory.memory_manager import load_memory, save_memory, clear_memory


# ── Scenario 1: First-time user ────────────────────────────────────────────────

class TestFirstTimeUser:

    def test_first_upload_no_history(self, sample_pdf, tmp_path, monkeypatch):
        """New user uploads first statement — insights runs successfully."""
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        parse = ParserAgent().run(str(sample_pdf))
        # Parser saves the session; InsightsAgent will load it as history (1 session).
        insights = InsightsAgent().run(parse["transactions"])

        assert parse["status"] == "success"
        assert insights["status"] == "success"
        assert "summary" in insights
        assert "recommendations" in insights

    def test_first_upload_saves_to_memory(self, sample_pdf, tmp_path, monkeypatch):
        """First upload should persist a session to memory."""
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        result = ParserAgent().run(str(sample_pdf))
        assert result["saved_to_memory"] is True
        assert len(load_memory(mem_file)["sessions"]) == 1

    def test_first_upload_recommendations_generated(self, sample_pdf, tmp_path, monkeypatch):
        """Even on first upload, recommendations should be produced."""
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        parse = ParserAgent().run(str(sample_pdf))
        insights = InsightsAgent().run(parse["transactions"])

        assert isinstance(insights["recommendations"], list)
        assert len(insights["recommendations"]) >= 1


# ── Scenario 2: Returning user with history ────────────────────────────────────

class TestReturningUser:

    def test_second_upload_loads_history(self, sample_pdf, tmp_path, monkeypatch):
        """Second upload should use first month as history."""
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        agent = ParserAgent()
        first = agent.run(str(sample_pdf))
        assert first["status"] == "success"

        second = agent.run(str(sample_pdf))
        insights = InsightsAgent().run(second["transactions"])

        assert insights["used_memory"] is True
        assert insights["comparison"]["status"] == "has_history"

    def test_history_accumulates_across_uploads(self, sample_pdf, tmp_path, monkeypatch):
        """Memory should grow with each successful upload."""
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        agent = ParserAgent()
        agent.run(str(sample_pdf))
        agent.run(str(sample_pdf))
        agent.run(str(sample_pdf))

        assert len(load_memory(mem_file)["sessions"]) == 3


# ── Scenario 3: User clears memory ────────────────────────────────────────────

class TestUserClearsMemory:

    def test_after_clear_next_upload_is_first_upload(self, sample_transactions, tmp_path, monkeypatch):
        """After clearing memory, InsightsAgent sees no history when memory is empty."""
        import memory.memory_manager as mm
        mem_file = tmp_path / "mem.json"
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", mem_file)

        from memory.memory_manager import save_memory
        save_memory({"month": "February", "year": 2026, "total_income": 100000.0,
                     "total_spend": 80000.0, "net_savings": 20000.0,
                     "savings_rate": 20.0, "categories": {}, "transaction_count": 10}, mem_file)
        clear_memory(mem_file)

        # After clearing, memory is empty — insights should see no history
        insights = InsightsAgent().run(sample_transactions)
        assert insights["comparison"]["status"] == "first_upload"
        assert insights["used_memory"] is False

    def test_clear_creates_backup_before_wipe(self, sample_summary, tmp_path):
        """Clearing memory must create a backup file."""
        mem_file = tmp_path / "mem.json"
        save_memory(sample_summary, mem_file)
        clear_memory(mem_file)
        assert mem_file.with_suffix(".json.backup").exists()


# ── Scenario 4: Bad file handling ─────────────────────────────────────────────

class TestBadFileScenarios:

    def test_nonexistent_file_returns_error(self):
        result = ParserAgent().run("/tmp/ghost_file_xyz.pdf")
        assert result["status"] == "error"
        assert result["saved_to_memory"] is False

    def test_error_does_not_break_subsequent_valid_run(self, sample_pdf, tmp_path, monkeypatch):
        """An error run should not prevent a subsequent valid run."""
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        agent = ParserAgent()
        bad = agent.run("/no/such/file.pdf")
        assert bad["status"] == "error"

        good = agent.run(str(sample_pdf))
        assert good["status"] == "success"
