"""Personal Finance Analyzer — Streamlit Application."""

import logging
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.insights_agent import InsightsAgent
from agents.parser_agent import ParserAgent
from memory.memory_manager import clear_memory, load_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar: hidden ── */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Hide Streamlit top header bar ── */
header[data-testid="stHeader"] { display: none !important; }

/* ── Container ── */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-bottom: 2rem !important;
    margin-top: 0 !important;
    max-width: 1200px;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 0.25rem;
}
.app-header-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0;
    line-height: 1.2;
}
.app-header-sub {
    font-size: 0.84rem;
    color: #64748B;
    margin: 3px 0 0;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #2563EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    color: #64748B !important;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1E293B !important;
    font-size: 1.35rem !important;
    font-weight: 700;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem;
}

/* ── Tabs — underline style ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 2px solid #E2E8F0;
    border-radius: 0;
    padding: 0;
    margin-bottom: 1.75rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 8px 22px 12px;
    font-weight: 500;
    font-size: 0.875rem;
    color: #64748B;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2563EB !important;
    font-weight: 600;
    border-bottom: 3px solid #2563EB !important;
    box-shadow: none;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2563EB;
    background: #F8FAFC !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #2563EB;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 2rem;
    letter-spacing: 0.02em;
    transition: background 0.15s;
    width: 100%;
}
.stButton > button[kind="primary"]:hover { background: #1D4ED8; }
.stButton > button[kind="secondary"] { border-radius: 8px; }

/* ── Settings card ── */
.settings-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 22px 18px;
}
.settings-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    padding-bottom: 10px;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 14px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] > div {
    border: 2px dashed #BFDBFE;
    border-radius: 12px;
    background: #F0F7FF;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #2563EB;
    background: #EFF6FF;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    box-shadow: none;
    background: #FAFBFC;
}
div[data-testid="stExpander"] summary {
    font-weight: 500;
    color: #475569;
}

/* ── DataFrames ── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94A3B8;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* ── Recommendation card ── */
.rec-card {
    display: flex;
    gap: 14px;
    background: #fff;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    align-items: flex-start;
}
.rec-num {
    min-width: 26px;
    height: 26px;
    background: #EFF6FF;
    color: #2563EB;
    font-weight: 700;
    font-size: 0.8rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.rec-text { color: #334155; font-size: 0.92rem; line-height: 1.65; }

/* ── Quick-win item ── */
.win-item {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #166534;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Savings callout ── */
.savings-callout {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 1.5rem;
}
.savings-callout-title { font-weight: 600; color: #1E40AF; font-size: 0.92rem; margin-bottom: 4px; }
.savings-callout-body  { color: #2563EB; font-size: 0.88rem; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4rem 2rem; }
.empty-state-icon { font-size: 2.6rem; display: block; margin-bottom: 1rem; }
.empty-state-title { font-size: 1rem; font-weight: 600; color: #475569; margin-bottom: 0.4rem; }
.empty-state-sub { font-size: 0.85rem; color: #94A3B8; }

/* ── Anomaly card ── */
.anomaly-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.anomaly-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.anomaly-merchant { font-weight: 600; color: #1E293B; font-size: 0.93rem; }
.anomaly-amount   { font-weight: 700; color: #92400E; font-size: 0.93rem; }
.anomaly-badge {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 4px;
    padding: 2px 7px;
    margin-right: 6px;
}
.anomaly-meta { font-size: 0.81rem; color: #64748B; margin-top: 4px; }

/* ── Parse result ── */
.parse-success {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 1rem 0;
}
.parse-success-title { font-weight: 600; color: #166534; font-size: 0.95rem; margin-bottom: 4px; }
.parse-success-sub   { font-size: 0.84rem; color: #4B7B5A; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"


def _save_upload(uploaded_file) -> str:
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    dest = uploads / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return str(dest)


def _empty_state(icon: str, title: str, sub: str) -> None:
    st.markdown(f"""
    <div class="empty-state">
        <span class="empty-state-icon">{icon}</span>
        <p class="empty-state-title">{title}</p>
        <p class="empty-state-sub">{sub}</p>
    </div>""", unsafe_allow_html=True)


def _section_label(text: str) -> None:
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "parsed_result":  None,
        "insights_result": None,
        "parser_agent":   None,
        "insights_agent": None,
        # settings (previously sidebar)
        "llm_provider":   "Claude (Recommended)",
        "llm_model":      "claude-sonnet-4-6",
        "api_key":        os.getenv("ANTHROPIC_API_KEY", ""),
        "analysis_depth": "Standard",
        "auto_save":      True,
        "show_details":   True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.parser_agent is None:
        st.session_state.parser_agent = ParserAgent()
    if st.session_state.insights_agent is None:
        st.session_state.insights_agent = InsightsAgent()


# ── Settings panel (inline, Upload tab) ───────────────────────────────────────

_MODEL_MAP = {
    "Claude (Recommended)": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "OpenAI":  ["gpt-4o", "gpt-4o-mini"],
    "Gemini":  ["gemini-1.5-pro", "gemini-1.5-flash"],
}


def _render_settings_panel() -> None:
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown('<p class="settings-title">⚙ Analysis Settings</p>', unsafe_allow_html=True)

    provider = st.radio(
        "AI Provider",
        list(_MODEL_MAP.keys()),
        key="llm_provider",
        horizontal=False,
    )

    st.selectbox("Model", _MODEL_MAP[provider], key="llm_model")

    st.text_input(
        "API Key",
        type="password",
        key="api_key",
        placeholder="Paste your API key…",
        help="Required only when using the LLM-assisted analysis features.",
    )

    st.divider()

    st.radio(
        "Analysis Depth",
        ["Quick (3 months)", "Standard", "Deep (full history)"],
        index=1,
        key="analysis_depth",
        help="Controls how many months of history are used for trend comparisons.",
    )

    st.checkbox("Auto-save results to memory", key="auto_save")
    st.checkbox("Show extracted transactions after parsing", key="show_details")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 1: Upload & Parse ──────────────────────────────────────────────────────

def render_upload_tab() -> None:
    col_upload, col_settings = st.columns([3, 2], gap="large")

    with col_settings:
        _render_settings_panel()

    with col_upload:
        _section_label("Bank Statement")
        uploaded_file = st.file_uploader(
            "Drop your PDF here or click to browse",
            type=["pdf"],
            label_visibility="collapsed",
            help="Supports IndusInd, HDFC, ICICI, Axis, and most Indian bank PDF statements.",
            key="pdf_uploader",
        )

        if uploaded_file is None:
            st.markdown("""
            <div style="color:#94A3B8; font-size:0.875rem; margin-top:0.5rem;">
                Supports IndusInd, HDFC, ICICI, Axis, and most Indian bank PDF statements.
            </div>""", unsafe_allow_html=True)
            return

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:10px 14px;
                    background:#F1F5F9; border-radius:8px; margin:10px 0;
                    font-size:0.875rem; color:#475569;">
            <span style="font-size:1.2rem;">📄</span>
            <span><strong>{uploaded_file.name}</strong> &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB</span>
        </div>""", unsafe_allow_html=True)

        if st.button("Parse Statement", type="primary", key="parse_btn"):
            with st.spinner("Extracting transactions…"):
                try:
                    filepath = _save_upload(uploaded_file)
                    result = st.session_state.parser_agent.run(filepath)
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    logger.exception("Upload tab parse error")
                    return

            if result["status"] == "error":
                st.error(f"Parse failed: {result.get('error', 'Unknown error')}")
                with st.expander("Debug: Raw PDF text (first 1 000 chars)"):
                    from tools.file_tools import read_pdf
                    raw = read_pdf(_save_upload(uploaded_file)) or ""
                    st.code(raw[:1000] if raw else "No text could be extracted.")
                return

            st.session_state.parsed_result = result
            st.markdown(f"""
            <div class="parse-success">
                <p class="parse-success-title">
                    ✓ &nbsp;{result['count']} transactions extracted
                </p>
                <p class="parse-success-sub">
                    Statement period: {result['month']} {result['year']}
                    &nbsp;·&nbsp; Switch to the <strong>Dashboard</strong> tab to view insights.
                </p>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions", result["count"])
            c2.metric("Month", result["month"])
            c3.metric("Year", result["year"])
            c4.metric("Saved to Memory", "Yes" if result.get("saved_to_memory") else "No")

            if st.session_state.get("show_details", True):
                _section_label("Extracted Transactions")
                st.dataframe(
                    pd.DataFrame(result["transactions"]),
                    use_container_width=True,
                    height=280,
                    column_config={
                        "date":        st.column_config.TextColumn("Date",        width="small"),
                        "merchant":    st.column_config.TextColumn("Merchant"),
                        "description": st.column_config.TextColumn("Description"),
                        "amount":      st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                        "type":        st.column_config.TextColumn("Type",        width="small"),
                        "category":    st.column_config.TextColumn("Category"),
                    },
                )

            with st.spinner("Generating insights with Claude…"):
                try:
                    insights = st.session_state.insights_agent.run(
                        result["transactions"],
                        api_key=st.session_state.get("api_key", ""),
                    )
                    st.session_state.insights_result = insights
                except Exception as exc:
                    logger.exception("Auto-insights failed after upload")
                    st.warning(f"Could not generate insights automatically: {exc}")


# ── Tab 2: Dashboard ───────────────────────────────────────────────────────────

def render_dashboard_tab() -> None:
    parsed   = st.session_state.parsed_result
    insights = st.session_state.insights_result

    if parsed is None:
        _empty_state("📊", "No data yet", "Upload and parse a bank statement to see your dashboard.")
        return

    if insights is None or insights.get("status") == "error":
        with st.spinner("Generating insights with Claude…"):
            try:
                insights = st.session_state.insights_agent.run(
                    parsed["transactions"],
                    api_key=st.session_state.get("api_key", ""),
                )
                st.session_state.insights_result = insights
            except Exception as exc:
                st.error(f"Failed to generate insights: {exc}")
                return

    if insights.get("status") == "error":
        st.error(f"Insights error: {insights.get('error')}")
        return

    summary = insights["summary"]
    month_label = f"{summary.get('month', '')} {summary.get('year', '')}".strip()

    # ── Period heading ──
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:baseline;
                margin-bottom:1.25rem;">
        <div>
            <span style="font-size:1.2rem; font-weight:700; color:#1E293B;">{month_label}</span>
            <span style="font-size:0.85rem; color:#64748B; margin-left:10px;">Overview</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income",  _fmt_inr(summary["total_income"]))
    c2.metric("Total Spend",   _fmt_inr(summary["total_spend"]))
    c3.metric("Net Savings",   _fmt_inr(summary["net_savings"]))
    c4.metric("Savings Rate",  f"{summary['savings_rate']:.1f}%")

    categories = {k: v for k, v in summary.get("categories", {}).items() if v > 0}
    if not categories:
        st.info("No spending data available for charts.")
        return

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Charts ──
    col_left, col_right = st.columns(2, gap="large")
    _PALETTE = ["#2563EB", "#0EA5E9", "#10B981", "#F59E0B",
                "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6"]

    with col_left:
        _section_label("Spending by Category")
        fig_pie = px.pie(
            values=list(categories.values()),
            names=list(categories.keys()),
            color_discrete_sequence=_PALETTE,
            hole=0.38,
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1E293B", size=12),
            legend=dict(font=dict(color="#1E293B", size=11), orientation="v"),
            showlegend=True,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    with col_right:
        _section_label("Category Breakdown")
        cats_sorted = dict(sorted(categories.items(), key=lambda x: x[1]))
        fig_bar = px.bar(
            x=list(cats_sorted.values()),
            y=list(cats_sorted.keys()),
            orientation="h",
            labels={"x": "Amount (₹)", "y": ""},
            color=list(cats_sorted.values()),
            color_continuous_scale=[[0, "#BFDBFE"], [0.5, "#3B82F6"], [1, "#1D4ED8"]],
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=0, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FAFBFE",
            font=dict(color="#1E293B", size=11),
            xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0",
                       tickprefix="₹", tickformat=","),
            yaxis=dict(gridcolor="#E2E8F0"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

    # ── Anomaly + Recommendations row ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_a, col_r = st.columns(2, gap="large")

    anomalies = insights.get("anomalies", [])
    with col_a:
        _section_label("Anomaly Status")
        if anomalies:
            st.warning(f"**{len(anomalies)} anomalies** detected — see the Anomalies tab for details.")
        else:
            st.success("No anomalies detected this month.")

    with col_r:
        used_claude = insights.get("used_claude", False)
        label = "Top Recommendations &nbsp;<span style='font-size:0.7rem;font-weight:600;" \
                "color:#1D4ED8;background:#EFF6FF;border-radius:10px;padding:1px 8px;" \
                "vertical-align:middle;'>Claude AI</span>" if used_claude else "Top Recommendations"
        st.markdown(f'<p class="section-label">{label}</p>', unsafe_allow_html=True)
        for i, rec in enumerate(insights.get("recommendations", [])[:3], 1):
            st.markdown(f"""
            <div class="rec-card" style="margin-bottom:8px;">
                <div class="rec-num">{i}</div>
                <div class="rec-text">{rec}</div>
            </div>""", unsafe_allow_html=True)


# ── Tab 3: History & Comparison ────────────────────────────────────────────────

def render_history_tab() -> None:
    memory   = load_memory()
    sessions = memory.get("sessions", [])

    if not sessions:
        _empty_state("📅", "No history yet",
                     "Parse at least one statement to start building your financial history.")
        return

    if len(sessions) < 2:
        st.info("Parse at least 2 months to unlock month-over-month comparison.")

    current  = sessions[-1]
    previous = sessions[-2] if len(sessions) >= 2 else None

    # ── MoM Comparison ──
    if previous:
        _section_label("Month-over-Month")
        col1, col2 = st.columns(2, gap="large")
        spend_delta = current["total_spend"] - previous["total_spend"]
        rate_delta  = current["savings_rate"] - previous["savings_rate"]

        with col1:
            st.markdown(f"**{current['month']} {current['year']}** — current")
            st.metric("Income",       _fmt_inr(current["total_income"]))
            st.metric("Spend",        _fmt_inr(current["total_spend"]))
            st.metric("Savings Rate", f"{current['savings_rate']:.1f}%")

        with col2:
            st.markdown(f"**{previous['month']} {previous['year']}** — previous")
            st.metric("Income", _fmt_inr(previous["total_income"]))
            st.metric("Spend",  _fmt_inr(current["total_spend"]),
                      delta=f"₹{spend_delta:+,.0f}", delta_color="inverse")
            st.metric("Savings Rate", f"{current['savings_rate']:.1f}%",
                      delta=f"{rate_delta:+.1f}%")

    # ── Trend chart ──
    if len(sessions) >= 2:
        _section_label("Spending Trend")
        df_trend = pd.DataFrame([
            {"Period": f"{s['month'][:3]} {s['year']}",
             "Spend":   s["total_spend"],
             "Income":  s["total_income"],
             "Savings": s["net_savings"]}
            for s in sessions
        ])
        fig = go.Figure()
        for label, color in [("Income", "#10B981"), ("Spend", "#EF4444"), ("Savings", "#2563EB")]:
            fig.add_trace(go.Scatter(
                x=df_trend["Period"], y=df_trend[label],
                name=label, line=dict(color=color, width=2.5),
                mode="lines+markers", marker=dict(size=6),
            ))
        fig.update_layout(
            xaxis_title="", yaxis_title="Amount (₹)",
            margin=dict(t=10, b=10, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFBFE",
            font=dict(color="#1E293B", size=12),
            xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#E2E8F0", tickprefix="₹", tickformat=","),
            legend=dict(font=dict(color="#1E293B"), orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key="trend_chart")

    # ── Category delta ──
    if previous and "categories" in current and "categories" in previous:
        _section_label("Category Changes")
        deltas = []
        for cat, curr_amt in current["categories"].items():
            prev_amt = previous["categories"].get(cat, 0)
            if prev_amt > 0:
                pct = (curr_amt - prev_amt) / prev_amt * 100
                deltas.append({"Category": cat,
                               "Previous (₹)": round(prev_amt, 0),
                               "Current (₹)":  round(curr_amt, 0),
                               "Change %":     round(pct, 1)})
        if deltas:
            df_delta = pd.DataFrame(deltas).sort_values("Change %", ascending=False)
            st.dataframe(df_delta, use_container_width=True)

    # ── All sessions ──
    _section_label("All Sessions")
    for s in reversed(sessions):
        label = f"{s['month']} {s['year']}  ·  {_fmt_inr(s['total_spend'])} spend"
        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            c1.metric("Income",       _fmt_inr(s["total_income"]))
            c2.metric("Spend",        _fmt_inr(s["total_spend"]))
            c3.metric("Savings Rate", f"{s['savings_rate']:.1f}%")


# ── Tab 4: Anomalies ───────────────────────────────────────────────────────────

def render_anomalies_tab() -> None:
    insights = st.session_state.insights_result
    if insights is None:
        _empty_state("⚠️", "No data yet", "Upload and parse a statement to detect anomalies.")
        return

    anomalies = insights.get("anomalies", [])

    if not anomalies:
        st.success("No anomalies detected — spending looks normal this month.")
        return

    # ── Summary metrics ──
    reason_counts = Counter(a.get("reason", "unknown") for a in anomalies)
    cols = st.columns(max(len(reason_counts), 1))
    for i, (reason, count) in enumerate(reason_counts.items()):
        cols[i].metric(reason.replace("_", " ").title(), count)

    # ── Anomaly cards ──
    _section_label(f"{len(anomalies)} Flagged Transactions")
    for a in anomalies:
        reason_label = a.get("reason", "unknown").replace("_", " ").title()
        tip = ""
        reason = a.get("reason", "")
        if "large" in reason:
            tip = "Consider whether this is a recurring expense or a one-off."
        elif "spike" in reason:
            tip = "This category is significantly above your 3-month average."
        elif "duplicate" in reason:
            tip = "Possible duplicate — verify with your bank."

        st.markdown(f"""
        <div class="anomaly-card">
            <div class="anomaly-top">
                <span class="anomaly-merchant">{a.get('merchant', 'Unknown')}</span>
                <span class="anomaly-amount">{_fmt_inr(a.get('amount', 0))}</span>
            </div>
            <div>
                <span class="anomaly-badge">{reason_label}</span>
                <span class="anomaly-meta">{a.get('date', '')} &nbsp;·&nbsp; {a.get('category', '')}</span>
            </div>
            {"<div style='font-size:0.82rem;color:#92400E;margin-top:8px;'>💡 " + tip + "</div>" if tip else ""}
        </div>""", unsafe_allow_html=True)

    # ── Raw table toggle ──
    with st.expander("View as table"):
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True)


# ── Tab 5: Recommendations ─────────────────────────────────────────────────────

def render_recommendations_tab() -> None:
    insights = st.session_state.insights_result
    if insights is None:
        _empty_state("💡", "No data yet", "Upload and parse a statement to generate recommendations.")
        return

    recommendations = insights.get("recommendations", [])
    summary         = insights.get("summary", {})
    used_claude     = insights.get("used_claude", False)
    rec_error       = insights.get("rec_error")

    # ── Source badge ──
    if used_claude:
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#EFF6FF;border:1px solid #BFDBFE;border-radius:20px;
                    padding:5px 14px;margin-bottom:1.25rem;">
            <span style="font-size:1rem;">✦</span>
            <span style="font-size:0.82rem;font-weight:600;color:#1D4ED8;">
                Powered by Claude AI
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        msg = f" &nbsp;·&nbsp; <span style='color:#94A3B8;font-size:0.78rem;'>{rec_error}</span>" if rec_error else ""
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#F8FAFC;border:1px solid #E2E8F0;border-radius:20px;
                    padding:5px 14px;margin-bottom:1.25rem;">
            <span style="font-size:1rem;">⚙</span>
            <span style="font-size:0.82rem;font-weight:600;color:#64748B;">
                Rule-based analysis{msg}
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Savings opportunity callout ──
    if summary:
        income       = summary.get("total_income", 0)
        current_rate = summary.get("savings_rate", 0)
        target_rate  = min(current_rate + 5, 30)
        potential    = (target_rate - current_rate) / 100 * income
        if potential > 0:
            st.markdown(f"""
            <div class="savings-callout">
                <p class="savings-callout-title">💰 Savings Opportunity</p>
                <p class="savings-callout-body">
                    Raising your savings rate from <strong>{current_rate:.1f}%</strong> to
                    <strong>{target_rate:.1f}%</strong> would free up an extra
                    <strong>{_fmt_inr(potential)} / month</strong>.
                </p>
            </div>""", unsafe_allow_html=True)

    # ── Personalised tips ──
    if recommendations:
        _section_label("Personalised Tips")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-num">{i}</div>
                <div class="rec-text">{rec}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No personalised recommendations generated.")

    # ── Quick wins ──
    _section_label("Quick Wins")
    for win in [
        ("🔍", "Review all active subscriptions and cancel unused ones"),
        ("🛒", "Switch to generic brands for 30% of grocery purchases"),
        ("🍱", "Meal-prep 2 days per week to reduce food-delivery spend"),
        ("📲", "Use UPI cashback offers for everyday transactions"),
    ]:
        st.markdown(f"""
        <div class="win-item">
            <span style="font-size:1.1rem;">{win[0]}</span>
            <span>{win[1]}</span>
        </div>""", unsafe_allow_html=True)


# ── Tab 6: Memory ──────────────────────────────────────────────────────────────

def render_memory_tab() -> None:
    memory   = load_memory()
    sessions = memory.get("sessions", [])

    # ── Summary metrics ──
    _section_label("Storage Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions Stored", len(sessions))
    if sessions:
        date_range = f"{sessions[0].get('month', '?')} – {sessions[-1].get('month', '?')}"
        avg_spend  = sum(s.get("total_spend", 0) for s in sessions) / len(sessions)
        c2.metric("Date Range",        date_range)
        c3.metric("Avg Monthly Spend", _fmt_inr(avg_spend))
    else:
        c2.metric("Date Range",        "—")
        c3.metric("Avg Monthly Spend", "—")

    if not sessions:
        _empty_state("🗄️", "Nothing stored yet", "Parse a statement to build your financial history.")
        return

    # ── Sessions table ──
    _section_label("Stored Sessions")
    display_cols = ["month", "year", "total_income", "total_spend", "net_savings", "savings_rate"]
    df = pd.DataFrame(sessions)
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available],
        use_container_width=True,
        column_config={
            "month":        st.column_config.TextColumn("Month"),
            "year":         st.column_config.NumberColumn("Year", format="%d"),
            "total_income": st.column_config.NumberColumn("Income (₹)",     format="₹%.0f"),
            "total_spend":  st.column_config.NumberColumn("Spend (₹)",      format="₹%.0f"),
            "net_savings":  st.column_config.NumberColumn("Net Savings (₹)", format="₹%.0f"),
            "savings_rate": st.column_config.NumberColumn("Savings Rate %",  format="%.1f%%"),
        },
    )

    # ── Export ──
    _section_label("Export")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download History (CSV)",
        data=csv,
        file_name="finance_history.csv",
        mime="text/csv",
        key="export_csv",
    )

    # ── Clear memory ──
    _section_label("Danger Zone")
    with st.expander("Clear all stored data"):
        st.warning("This is permanent. A backup file will be created automatically.")
        confirm = st.checkbox("I understand — permanently delete all stored sessions",
                              key="confirm_clear")
        if st.button("Clear All Memory", disabled=not confirm, key="clear_btn"):
            if clear_memory():
                st.success("Memory cleared. Backup saved.")
                st.rerun()
            else:
                st.error("Failed to clear memory — check file permissions.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # App header
    st.markdown("""
    <div class="app-header">
        <div>
            <p class="app-header-title">💰 Personal Finance Analyzer</p>
            <p class="app-header-sub">
                Upload your bank statement and get intelligent insights into your spending.
            </p>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Upload & Parse",
        "Dashboard",
        "History",
        "Anomalies",
        "Recommendations",
        "Memory",
    ])

    with tab1: render_upload_tab()
    with tab2: render_dashboard_tab()
    with tab3: render_history_tab()
    with tab4: render_anomalies_tab()
    with tab5: render_recommendations_tab()
    with tab6: render_memory_tab()


if __name__ == "__main__":
    main()
