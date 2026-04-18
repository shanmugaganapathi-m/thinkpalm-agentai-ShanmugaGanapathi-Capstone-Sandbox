"""
UI tests for the Streamlit application (Phase 5).

Uses streamlit.testing.v1.AppTest to run the app headlessly and inspect
rendered elements, session state, and widget interactions.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = "app.py"
_TIMEOUT = 30  # seconds


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fresh_app() -> AppTest:
    """Return a new AppTest instance with the app loaded but not yet run."""
    return AppTest.from_file(APP_FILE, default_timeout=_TIMEOUT)


def _run_app(**session_overrides) -> AppTest:
    """Run the app, optionally seeding session state before execution."""
    at = _fresh_app()
    for key, val in session_overrides.items():
        at.session_state[key] = val
    at.run()
    return at


# ── App loads ──────────────────────────────────────────────────────────────────

class TestAppLoads:

    def test_app_runs_without_exception(self):
        at = _run_app()
        assert not at.exception

    def test_title_contains_finance(self):
        at = _run_app()
        titles = [t.value for t in at.title]
        assert any("Finance" in t for t in titles)

    def test_session_state_parsed_result_initialized(self):
        at = _run_app()
        assert "parsed_result" in at.session_state
        assert at.session_state["parsed_result"] is None

    def test_session_state_insights_result_initialized(self):
        at = _run_app()
        assert "insights_result" in at.session_state
        assert at.session_state["insights_result"] is None

    def test_parser_agent_cached_in_session(self):
        at = _run_app()
        assert "parser_agent" in at.session_state
        assert at.session_state["parser_agent"] is not None

    def test_insights_agent_cached_in_session(self):
        at = _run_app()
        assert "insights_agent" in at.session_state
        assert at.session_state["insights_agent"] is not None

    def test_llm_config_stored_in_session(self):
        at = _run_app()
        assert "llm_config" in at.session_state
        cfg = at.session_state["llm_config"]
        assert "provider" in cfg
        assert "model" in cfg
        assert "depth" in cfg


# ── Sidebar ────────────────────────────────────────────────────────────────────

class TestSidebar:

    def test_llm_provider_radio_present(self):
        at = _run_app()
        keys = [r.key for r in at.radio]
        assert "llm_provider" in keys

    def test_model_selectbox_present(self):
        at = _run_app()
        keys = [s.key for s in at.selectbox]
        assert "llm_model" in keys

    def test_analysis_depth_radio_present(self):
        at = _run_app()
        keys = [r.key for r in at.radio]
        assert "analysis_depth" in keys

    def test_auto_save_checkbox_present(self):
        at = _run_app()
        keys = [c.key for c in at.checkbox]
        assert "auto_save" in keys

    def test_show_details_checkbox_present(self):
        at = _run_app()
        keys = [c.key for c in at.checkbox]
        assert "show_details" in keys

    def test_api_key_input_present(self):
        at = _run_app()
        keys = [t.key for t in at.text_input]
        assert "api_key" in keys

    def test_claude_selected_by_default(self):
        at = _run_app()
        radios = {r.key: r for r in at.radio}
        assert radios["llm_provider"].value == "Claude (Recommended)"


# ── Upload tab ─────────────────────────────────────────────────────────────────

class TestUploadTab:

    def test_no_file_shows_info_message(self):
        at = _run_app()
        info_texts = [i.value for i in at.info]
        assert any("Upload" in t or "upload" in t for t in info_texts)

    def test_upload_header_rendered(self):
        at = _run_app()
        headers = [h.value for h in at.header]
        assert any("Upload" in h for h in headers)

    def test_parse_button_absent_without_file(self):
        # parse_btn is conditionally rendered only after a file is uploaded
        at = _run_app()
        keys = [b.key for b in at.button]
        assert "parse_btn" not in keys  # expected: button hidden until file chosen


# ── Dashboard tab ──────────────────────────────────────────────────────────────

class TestDashboardTab:

    def test_dashboard_shows_warning_without_data(self):
        at = _run_app()
        warnings = [w.value for w in at.warning]
        assert any("parse" in w.lower() or "Upload" in w for w in warnings)

    def test_dashboard_with_mock_data_no_exception(self, sample_transactions, sample_summary):
        mock_insights = {
            "status": "success",
            "summary": sample_summary,
            "comparison": {"status": "first_upload", "deltas": {}},
            "anomalies": [],
            "recommendations": [
                "Cut food delivery spend",
                "Review subscriptions",
                "Increase SIP contribution",
            ],
            "used_memory": False,
            "history_months": 0,
        }
        mock_parsed = {
            "status": "success",
            "transactions": sample_transactions,
            "count": 3,
            "month": "March",
            "year": 2026,
            "saved_to_memory": True,
        }
        at = _run_app(parsed_result=mock_parsed, insights_result=mock_insights)
        assert not at.exception

    def test_dashboard_summary_metrics_rendered(self, sample_transactions, sample_summary):
        mock_insights = {
            "status": "success",
            "summary": sample_summary,
            "comparison": {"status": "first_upload", "deltas": {}},
            "anomalies": [],
            "recommendations": ["Tip one", "Tip two", "Tip three"],
            "used_memory": False,
            "history_months": 0,
        }
        mock_parsed = {
            "status": "success",
            "transactions": sample_transactions,
            "count": 3,
            "month": "March",
            "year": 2026,
            "saved_to_memory": True,
        }
        at = _run_app(parsed_result=mock_parsed, insights_result=mock_insights)
        metric_labels = [m.label for m in at.metric]
        assert any("Income" in lbl for lbl in metric_labels)
        assert any("Spend" in lbl for lbl in metric_labels)
        assert any("Savings" in lbl for lbl in metric_labels)


# ── History tab ────────────────────────────────────────────────────────────────

class TestHistoryTab:

    def test_history_no_sessions_shows_info(self, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")
        at = _run_app()
        assert not at.exception
        info_texts = [i.value for i in at.info]
        assert any("history" in t.lower() or "parse" in t.lower() for t in info_texts)


# ── Anomalies tab ──────────────────────────────────────────────────────────────

class TestAnomaliesTab:

    def test_anomalies_tab_no_insights_shows_warning(self):
        at = _run_app()
        warnings = [w.value for w in at.warning]
        assert any("insight" in w.lower() or "Upload" in w for w in warnings)

    def test_anomalies_tab_no_anomalies_shows_success(self, sample_transactions, sample_summary):
        mock_insights = {
            "status": "success",
            "summary": sample_summary,
            "comparison": {"status": "first_upload", "deltas": {}},
            "anomalies": [],
            "recommendations": ["Tip A", "Tip B", "Tip C"],
            "used_memory": False,
            "history_months": 0,
        }
        at = _run_app(insights_result=mock_insights)
        assert not at.exception
        success_texts = [s.value for s in at.success]
        assert any("anomal" in t.lower() for t in success_texts)


# ── Recommendations tab ────────────────────────────────────────────────────────

class TestRecommendationsTab:

    def test_recommendations_no_insights_shows_warning(self):
        at = _run_app()
        warnings = [w.value for w in at.warning]
        assert any("insight" in w.lower() or "Upload" in w for w in warnings)

    def test_recommendations_renders_with_data(self, sample_summary):
        mock_insights = {
            "status": "success",
            "summary": sample_summary,
            "comparison": {"status": "first_upload", "deltas": {}},
            "anomalies": [],
            "recommendations": [
                "Reduce food delivery by cooking at home 3x/week.",
                "Cancel two unused streaming subscriptions.",
                "Increase SIP by ₹2,000/month.",
            ],
            "used_memory": False,
            "history_months": 0,
        }
        at = _run_app(insights_result=mock_insights)
        assert not at.exception


# ── Memory tab ─────────────────────────────────────────────────────────────────

class TestMemoryTab:

    def test_memory_tab_no_exception(self, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")
        at = _run_app()
        assert not at.exception

    def test_memory_sessions_stored_metric(self, tmp_path, monkeypatch):
        import memory.memory_manager as mm
        monkeypatch.setattr(mm, "DEFAULT_MEMORY_FILE", tmp_path / "mem.json")
        at = _run_app()
        metric_labels = [m.label for m in at.metric]
        assert any("Session" in lbl for lbl in metric_labels)

    def test_clear_button_present(self):
        at = _run_app()
        keys = [b.key for b in at.button]
        assert "clear_btn" in keys

    def test_confirm_clear_checkbox_present(self):
        at = _run_app()
        keys = [c.key for c in at.checkbox]
        assert "confirm_clear" in keys


# ── Helpers (unit tests, no Streamlit context needed) ─────────────────────────

class TestHelpers:

    def test_fmt_inr_contains_rupee_symbol(self):
        from app import _fmt_inr
        assert _fmt_inr(132000.0).startswith("₹")

    def test_fmt_inr_formats_number(self):
        from app import _fmt_inr
        result = _fmt_inr(132000.0)
        assert "132" in result
        assert "000" in result

    def test_fmt_inr_zero(self):
        from app import _fmt_inr
        assert _fmt_inr(0.0) == "₹0"

    def test_save_upload_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from app import _save_upload

        class _MockUpload:
            name = "statement.pdf"
            def getbuffer(self):
                return b"%PDF-1.4 mock content"

        path = _save_upload(_MockUpload())
        assert Path(path).exists()
        assert Path(path).read_bytes() == b"%PDF-1.4 mock content"

    def test_save_upload_creates_uploads_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from app import _save_upload

        class _MockUpload:
            name = "x.pdf"
            def getbuffer(self):
                return b"data"

        _save_upload(_MockUpload())
        assert (tmp_path / "uploads").is_dir()
