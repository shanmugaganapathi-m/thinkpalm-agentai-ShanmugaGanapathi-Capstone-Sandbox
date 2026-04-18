"""
Insights Agent — Analyzes spending patterns and provides recommendations.

Agent 2 of the Personal Finance Analyzer.
Uses LangChain with Claude to orchestrate analysis tools.
Phase 4: Auto-loads historical data from memory when history is not supplied.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from memory.memory_manager import load_memory
from tools.analysis_tools import calculate_monthly_summary, compare_with_history
from tools.anomaly_tools import detect_anomalies
from tools.recommendation_tools import generate_recommendations

logger = logging.getLogger(__name__)

_HISTORY_WINDOW = 3  # months of history to use for comparisons

_SYSTEM_PROMPT = (
    "You are a personal finance advisor. Analyze spending patterns, detect anomalies, "
    "and generate actionable budget recommendations from transaction data. "
    "Tool order: calculate_monthly_summary → compare_with_history → "
    "detect_anomalies → generate_recommendations."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class InsightsAgent:
    """LangChain-backed agent that analyzes transactions and generates financial insights."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Initialize the Insights Agent.

        Args:
            llm: Optional LangChain LLM for testing/injection. Uses ChatAnthropic if None.
        """
        self.name = "Insights Agent"
        self._llm = llm
        self._executor: Optional[AgentExecutor] = None
        logger.info("%s initialized", self.name)

    # ------------------------------------------------------------------
    # LangChain executor (lazy-built, available for LLM-based features)
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[Tool]:
        """Wrap Phase-2 analysis tools as LangChain Tool objects."""
        return [
            Tool(
                name="calculate_monthly_summary",
                func=calculate_monthly_summary,
                description=(
                    "Aggregate transactions into a monthly summary with income, spend, "
                    "savings rate, and per-category totals. Input: transactions list."
                ),
            ),
            Tool(
                name="compare_with_history",
                func=lambda current: compare_with_history(current, []),
                description=(
                    "Compare the current month summary against historical averages and "
                    "flag categories with >10% change. Input: current_summary dict."
                ),
            ),
            Tool(
                name="detect_anomalies",
                func=detect_anomalies,
                description=(
                    "Detect category spikes, large single transactions (>₹10k), and "
                    "duplicate charges. Input: transactions list."
                ),
            ),
            Tool(
                name="generate_recommendations",
                func=lambda summary: generate_recommendations(summary, []),
                description=(
                    "Generate 3-5 actionable budget tips from a monthly summary. "
                    "Input: summary dict."
                ),
            ),
            Tool(
                name="load_memory",
                func=lambda _: str(load_memory()),
                description="Load prior session history for context. Pass any string as input.",
            ),
        ]

    def _get_executor(self) -> AgentExecutor:
        """Build and cache the LangChain AgentExecutor (lazy init)."""
        if self._executor is None:
            llm = self._llm or ChatAnthropic(
                model="claude-sonnet-4-6",
                temperature=0,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            tools = self._build_tools()
            agent = create_tool_calling_agent(llm, tools, _PROMPT)
            self._executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=15,
                handle_parsing_errors=True,
            )
            logger.debug("InsightsAgent executor built")
        return self._executor

    # ------------------------------------------------------------------
    # Main pipeline (deterministic — no LLM call required)
    # ------------------------------------------------------------------

    def run(
        self,
        transactions: List[Dict],
        history: Optional[List[Dict]] = None,
        api_key: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze transactions and return comprehensive financial insights.

        When ``history`` is ``None``, the last 3 sessions are loaded
        automatically from persistent memory (Phase 4 integration).

        Execution flow (ReAct):
          Thought      → "I need to analyze spending patterns"
          Action       → load_memory() [if history not supplied]
          Action       → calculate_monthly_summary(transactions)
          Observation  → {total_income, total_spend, savings_rate, categories}
          Action       → compare_with_history(summary, history)
          Observation  → {status, deltas}
          Action       → detect_anomalies(transactions, history)
          Observation  → [{date, merchant, amount, reason}, ...]
          Action       → generate_recommendations(summary, anomalies)
          Answer       → {status, summary, comparison, anomalies, recommendations,
                          used_memory, history_months}

        Args:
            transactions: Normalized transaction list from ParserAgent.
            history: Prior-month summaries for comparison. When ``None``,
                     loaded from persistent memory automatically.

        Returns:
            On success: {"status": "success", "summary": {...},
                         "comparison": {...}, "anomalies": [...],
                         "recommendations": [...],
                         "used_memory": bool, "history_months": int}
            On error:   {"status": "error", "error": str, "used_memory": False}
        """
        if not transactions:
            return {
                "status": "error",
                "error": "No transactions provided",
                "used_memory": False,
            }

        used_memory = False

        # Phase 4: auto-load history from memory when not supplied
        if history is None:
            try:
                mem = load_memory()
                history = mem.get("sessions", [])[-_HISTORY_WINDOW:]
                used_memory = len(history) > 0
                logger.info(
                    "InsightsAgent loaded %d session(s) from memory", len(history)
                )
            except Exception as exc:
                logger.warning("Could not load memory — proceeding without history: %s", exc)
                history = []
        else:
            used_memory = len(history) > 0

        history_months = len(history)

        logger.debug(
            "InsightsAgent.run() | transactions=%d history=%d used_memory=%s",
            len(transactions),
            history_months,
            used_memory,
        )

        try:
            logger.debug("Action: calculate_monthly_summary")
            summary = calculate_monthly_summary(transactions)
            logger.debug(
                "Observation: income=%.0f spend=%.0f savings_rate=%.1f%%",
                summary.get("total_income", 0),
                summary.get("total_spend", 0),
                summary.get("savings_rate", 0),
            )

            logger.debug("Action: compare_with_history (sessions=%d)", history_months)
            comparison = compare_with_history(summary, history)

            logger.debug("Action: detect_anomalies")
            anomalies = detect_anomalies(transactions, history)
            logger.debug("Observation: %d anomalies detected", len(anomalies))

            logger.debug("Action: generate_recommendations")
            rec_result = generate_recommendations(
                summary, anomalies,
                api_key=api_key,
                history=history,
            )
            recommendations = rec_result["items"]
            used_claude     = rec_result["used_claude"]
            rec_error       = rec_result.get("error")
            logger.debug(
                "Observation: %d recommendations (claude=%s)", len(recommendations), used_claude
            )

            return {
                "status": "success",
                "summary": summary,
                "comparison": comparison,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "used_claude": used_claude,
                "rec_error": rec_error,
                "used_memory": used_memory,
                "history_months": history_months,
            }

        except Exception as exc:
            logger.exception("InsightsAgent.run() failed")
            return {"status": "error", "error": str(exc), "used_memory": used_memory}

    def analyze(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Alias for run() — backward-compatible entry point."""
        return self.run(transactions)


if __name__ == "__main__":
    agent = InsightsAgent()
    print(f"{agent.name} ready")
