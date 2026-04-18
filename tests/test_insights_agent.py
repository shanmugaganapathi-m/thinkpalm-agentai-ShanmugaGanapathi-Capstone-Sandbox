"""Unit tests for InsightsAgent (Phase 3)."""

import pytest

from agents.insights_agent import InsightsAgent


class TestInsightsAgent:
    """Tests for InsightsAgent initialization, happy path, error handling, and edge cases."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_insights_agent_initialization(self):
        agent = InsightsAgent()
        assert agent.name == "Insights Agent"

    def test_insights_agent_tools_built(self):
        agent = InsightsAgent()
        tools = agent._build_tools()
        tool_names = [t.name for t in tools]
        assert "calculate_monthly_summary" in tool_names
        assert "compare_with_history" in tool_names
        assert "detect_anomalies" in tool_names
        assert "generate_recommendations" in tool_names
        assert "load_memory" in tool_names

    def test_insights_agent_accepts_llm_injection(self):
        fake_llm = object()
        agent = InsightsAgent(llm=fake_llm)
        assert agent._llm is fake_llm

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_insights_agent_summary_calculation(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions)
        assert result["status"] == "success"
        summary = result["summary"]
        assert summary["total_income"] == 132000.0
        assert summary["total_spend"] == pytest.approx(28900.0, abs=1.0)
        assert "categories" in summary

    def test_insights_agent_comparison_with_history(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions)
        assert result["status"] == "success"
        assert "comparison" in result
        assert "status" in result["comparison"]

    def test_insights_agent_anomaly_detection(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions)
        assert result["status"] == "success"
        assert isinstance(result["anomalies"], list)

    def test_insights_agent_recommendations_generation(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions)
        assert result["status"] == "success"
        recs = result["recommendations"]
        assert isinstance(recs, list)
        assert len(recs) >= 1
        for rec in recs:
            assert isinstance(rec, str)

    def test_insights_agent_recommendations_full_data(self, sample_summary):
        """Full monthly summary should produce 3-5 recommendations."""
        from tools.recommendation_tools import generate_recommendations
        recs = generate_recommendations(sample_summary, [])
        assert 3 <= len(recs) <= 5

    def test_insights_agent_all_keys_present(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions)
        assert result["status"] == "success"
        for key in ("summary", "comparison", "anomalies", "recommendations"):
            assert key in result

    # ------------------------------------------------------------------
    # Edge cases — history handling
    # ------------------------------------------------------------------

    def test_insights_agent_no_history(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.run(sample_transactions, history=[])  # explicit empty = first upload
        assert result["status"] == "success"
        assert result["comparison"]["status"] == "first_upload"

    def test_insights_agent_with_history(self, sample_transactions, sample_summary):
        agent = InsightsAgent()
        result = agent.run(sample_transactions, history=[sample_summary])
        assert result["status"] == "success"
        assert result["comparison"]["status"] == "has_history"

    def test_insights_agent_no_anomalies_small_transactions(self):
        small_txns = [
            {"date": "01-Jan-2026", "merchant": "Coffee", "description": "CAFE COFFEE",
             "amount": 100.0, "type": "expense", "category": "Food"},
            {"date": "02-Jan-2026", "merchant": "Salary", "description": "Salary Credit",
             "amount": 50000.0, "type": "income", "category": "Income"},
        ]
        agent = InsightsAgent()
        result = agent.run(small_txns)
        assert result["status"] == "success"
        large_anomalies = [
            a for a in result["anomalies"] if a.get("reason") == "large_transaction_>10k"
        ]
        assert len(large_anomalies) == 0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_insights_agent_error_on_empty_transactions(self):
        agent = InsightsAgent()
        result = agent.run([])
        assert result["status"] == "error"
        assert "error" in result

    def test_insights_agent_error_on_none_transactions(self):
        agent = InsightsAgent()
        result = agent.run(None)
        assert result["status"] == "error"

    # ------------------------------------------------------------------
    # Alias
    # ------------------------------------------------------------------

    def test_insights_agent_analyze_alias(self, sample_transactions):
        agent = InsightsAgent()
        result = agent.analyze(sample_transactions)
        assert result["status"] == "success"
