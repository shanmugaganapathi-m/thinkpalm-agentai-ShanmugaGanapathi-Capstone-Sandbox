"""
Performance benchmark tests (Phase 6).

Uses time.time() to assert that critical paths complete within acceptable limits.
These tests validate that no regressions slow down the pipeline.
"""

import time
import pytest

from tools.parser_tools import normalize_transactions, categorize_transaction
from tools.analysis_tools import calculate_monthly_summary, compare_with_history
from tools.anomaly_tools import detect_anomalies
from tools.recommendation_tools import generate_recommendations
from memory.memory_manager import load_memory, save_memory, clear_memory


# ── Tool-level benchmarks ──────────────────────────────────────────────────────

class TestToolPerformance:

    def test_categorize_transaction_under_1ms(self):
        start = time.time()
        for _ in range(1000):
            categorize_transaction("SWIGGY FOOD ORDER 450.00")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"1000 categorizations took {elapsed:.3f}s (expected <1s)"

    def test_normalize_transactions_under_500ms(self):
        raw = "\n".join(
            f"0{i % 28 + 1}-Mar-2026 MERCHANT{i} TXN DESC {100 + i}.00 {5000 - i}.00"
            for i in range(100)
        )
        start = time.time()
        normalize_transactions(raw)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"normalize_transactions took {elapsed:.3f}s (expected <0.5s)"

    def test_calculate_monthly_summary_under_100ms(self, sample_transactions):
        start = time.time()
        for _ in range(100):
            calculate_monthly_summary(sample_transactions)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"100 summary calculations took {elapsed:.3f}s"

    def test_compare_with_history_under_100ms(self, sample_summary):
        history = [sample_summary] * 3
        current = dict(sample_summary)
        start = time.time()
        for _ in range(100):
            compare_with_history(current, history)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"100 history comparisons took {elapsed:.3f}s"

    def test_detect_anomalies_under_200ms(self, sample_transactions, sample_summary):
        history = [sample_summary]
        start = time.time()
        for _ in range(100):
            detect_anomalies(sample_transactions, history)
        elapsed = time.time() - start
        assert elapsed < 0.2, f"100 anomaly detections took {elapsed:.3f}s"

    def test_generate_recommendations_under_100ms(self, sample_summary):
        start = time.time()
        for _ in range(100):
            generate_recommendations(sample_summary, [])
        elapsed = time.time() - start
        assert elapsed < 0.1, f"100 recommendation generations took {elapsed:.3f}s"


# ── Memory performance ─────────────────────────────────────────────────────────

class TestMemoryPerformance:

    def test_save_memory_under_500ms(self, sample_summary, tmp_path):
        f = tmp_path / "perf_mem.json"
        start = time.time()
        for i in range(10):
            save_memory({**sample_summary, "month": f"Month{i}"}, f)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"10 memory saves took {elapsed:.3f}s"

    def test_load_memory_under_500ms(self, sample_summary, tmp_path):
        f = tmp_path / "perf_mem.json"
        for i in range(5):
            save_memory({**sample_summary, "month": f"Month{i}"}, f)
        start = time.time()
        for _ in range(50):
            load_memory(f)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"50 memory loads took {elapsed:.3f}s"

    def test_clear_memory_under_100ms(self, sample_summary, tmp_path):
        f = tmp_path / "perf_mem.json"
        save_memory(sample_summary, f)
        start = time.time()
        clear_memory(f)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"clear_memory took {elapsed:.3f}s"


# ── Agent performance ──────────────────────────────────────────────────────────

class TestAgentPerformance:

    def test_insights_agent_run_under_2s(self, sample_transactions, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        from agents.insights_agent import InsightsAgent
        agent = InsightsAgent()
        start = time.time()
        result = agent.run(sample_transactions)
        elapsed = time.time() - start
        assert result["status"] == "success"
        assert elapsed < 2.0, f"InsightsAgent.run() took {elapsed:.3f}s (expected <2s)"

    def test_parser_agent_pdf_parse_under_5s(self, sample_pdf, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")

        from agents.parser_agent import ParserAgent
        agent = ParserAgent()
        start = time.time()
        result = agent.run(str(sample_pdf))
        elapsed = time.time() - start
        assert result["status"] == "success"
        assert elapsed < 5.0, f"ParserAgent.run() took {elapsed:.3f}s (expected <5s)"
