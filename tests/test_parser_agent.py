"""Unit tests for ParserAgent (Phase 3)."""

import pytest
from pathlib import Path

from agents.parser_agent import ParserAgent


class TestParserAgent:
    """Tests for ParserAgent initialization, happy path, error handling, and edge cases."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_parser_agent_initialization(self):
        agent = ParserAgent()
        assert agent.name == "Parser Agent"

    def test_parser_agent_tools_built(self):
        agent = ParserAgent()
        tools = agent._build_tools()
        tool_names = [t.name for t in tools]
        assert "validate_pdf" in tool_names
        assert "read_pdf" in tool_names
        assert "normalize_transactions" in tool_names
        assert "categorize_transaction" in tool_names

    def test_parser_agent_accepts_llm_injection(self):
        fake_llm = object()
        agent = ParserAgent(llm=fake_llm)
        assert agent._llm is fake_llm

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_parser_agent_pdf_upload_success(self, sample_pdf):
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"

    def test_parser_agent_transaction_count(self, sample_pdf):
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"
        assert result["count"] == len(result["transactions"])
        assert result["count"] > 0

    def test_parser_agent_categories_assigned(self, sample_pdf):
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"
        for txn in result["transactions"]:
            assert "category" in txn
            assert txn["category"] != ""

    def test_parser_agent_amount_calculation(self, sample_pdf):
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"
        for txn in result["transactions"]:
            assert isinstance(txn["amount"], float)
            assert txn["amount"] > 0

    def test_parser_agent_month_year_extracted(self, sample_pdf):
        agent = ParserAgent()
        result = agent.run(str(sample_pdf))
        assert result["status"] == "success"
        assert result["month"] == "March"
        assert result["year"] == 2026

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_parser_agent_invalid_filepath(self):
        agent = ParserAgent()
        result = agent.run("/nonexistent/path/statement.pdf")
        assert result["status"] == "error"
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_parser_agent_malformed_pdf(self, tmp_path):
        bad_file = tmp_path / "fake.pdf"
        bad_file.write_text("this is not a real pdf")
        agent = ParserAgent()
        result = agent.run(str(bad_file))
        assert result["status"] == "error"
        assert "error" in result

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_parser_agent_empty_pdf(self, tmp_path):
        import fitz
        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()
        agent = ParserAgent()
        result = agent.run(str(pdf_path))
        assert result["status"] == "error"
        assert "error" in result

    def test_parser_agent_process_pdf_alias(self, sample_pdf):
        agent = ParserAgent()
        result = agent.process_pdf(str(sample_pdf))
        assert result["status"] == "success"

    def test_parser_agent_extract_period_unknown_on_bad_date(self):
        agent = ParserAgent()
        month, year = agent._extract_period([{"date": "bad-date"}])
        assert month == "Unknown"
        assert year == 0

    def test_parser_agent_extract_period_empty_list(self):
        agent = ParserAgent()
        month, year = agent._extract_period([])
        assert month == "Unknown"
        assert year == 0
