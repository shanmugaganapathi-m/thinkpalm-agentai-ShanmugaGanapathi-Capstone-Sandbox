"""
Parser Agent — Ingests bank statements and extracts transactions.

Agent 1 of the Personal Finance Analyzer.
Uses LangChain with Claude to orchestrate PDF parsing tools.
Phase 4: Saves each parsed session to persistent memory.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from memory.memory_manager import save_memory
from tools.analysis_tools import calculate_monthly_summary
from tools.file_tools import read_pdf, validate_pdf
from tools.parser_tools import categorize_transaction, normalize_transactions

logger = logging.getLogger(__name__)

_MONTHS: Dict[str, str] = {
    "Jan": "January", "Feb": "February", "Mar": "March",
    "Apr": "April",   "May": "May",      "Jun": "June",
    "Jul": "July",    "Aug": "August",   "Sep": "September",
    "Oct": "October", "Nov": "November", "Dec": "December",
}

_SYSTEM_PROMPT = (
    "You are a financial document parser. Extract and structure transactions "
    "from PDF bank statements using the available tools. "
    "Tool order: validate_pdf → read_pdf → normalize_transactions. "
    "Return all transactions found in the document."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class ParserAgent:
    """LangChain-backed agent that parses PDF bank statements into structured transactions."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Initialize the Parser Agent.

        Args:
            llm: Optional LangChain LLM for testing/injection. Uses ChatAnthropic if None.
        """
        self.name = "Parser Agent"
        self._llm = llm
        self._executor: Optional[AgentExecutor] = None
        logger.info("%s initialized", self.name)

    # ------------------------------------------------------------------
    # LangChain executor (lazy-built, available for LLM-based features)
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[Tool]:
        """Wrap Phase-2 tools as LangChain Tool objects."""
        return [
            Tool(
                name="validate_pdf",
                func=validate_pdf,
                description=(
                    "Validate that a filepath points to a readable PDF. "
                    "Input: filepath string. Returns True/False."
                ),
            ),
            Tool(
                name="read_pdf",
                func=read_pdf,
                description=(
                    "Extract all raw text from a PDF bank statement. "
                    "Input: filepath string. Returns text string or None."
                ),
            ),
            Tool(
                name="normalize_transactions",
                func=normalize_transactions,
                description=(
                    "Parse raw bank-statement text into a list of structured "
                    "transaction dicts. Input: raw text string."
                ),
            ),
            Tool(
                name="categorize_transaction",
                func=categorize_transaction,
                description=(
                    "Return the spending category for a transaction description. "
                    "Input: description string."
                ),
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
                max_iterations=10,
                handle_parsing_errors=True,
            )
            logger.debug("ParserAgent executor built")
        return self._executor

    # ------------------------------------------------------------------
    # Main pipeline (deterministic — no LLM call required)
    # ------------------------------------------------------------------

    def run(self, filepath: str) -> Dict[str, Any]:
        """
        Parse a PDF bank statement, return structured transactions, and
        persist the session summary to memory.

        Execution flow (ReAct):
          Thought  → "I need to extract transactions from a PDF"
          Action   → validate_pdf(filepath)
          Action   → read_pdf(filepath)
          Action   → normalize_transactions(raw_text)
          Action   → save_memory(session_summary)
          Answer   → {status, transactions, count, month, year, saved_to_memory}

        Args:
            filepath: Path to the PDF bank statement.

        Returns:
            On success: {"status": "success", "transactions": [...],
                         "count": int, "month": str, "year": int,
                         "saved_to_memory": bool}
            On error:   {"status": "error", "error": str, "saved_to_memory": False}
        """
        logger.debug("ParserAgent.run() | filepath=%s", filepath)

        # --- pre-checks (no LLM needed) ---
        if not os.path.exists(filepath):
            logger.error("File not found: %s", filepath)
            return {
                "status": "error",
                "error": f"File not found: {filepath}",
                "saved_to_memory": False,
            }

        if not validate_pdf(filepath):
            logger.error("Invalid PDF: %s", filepath)
            return {
                "status": "error",
                "error": f"Invalid or unreadable PDF: {filepath}",
                "saved_to_memory": False,
            }

        try:
            logger.debug("Action: read_pdf(%s)", filepath)
            raw_text = read_pdf(filepath)

            if not raw_text or not raw_text.strip():
                return {
                    "status": "error",
                    "error": "PDF is empty or contains no extractable text",
                    "saved_to_memory": False,
                }

            logger.debug("Action: normalize_transactions (chars=%d)", len(raw_text))
            transactions = normalize_transactions(raw_text)

            if not transactions:
                sample = raw_text[:800].replace("\n", " | ")
                return {
                    "status": "error",
                    "error": (
                        "No transactions found in PDF. "
                        "The parser looks for date patterns like DD/MM/YYYY, DD-MM-YYYY, "
                        "DD-Mon-YYYY, or YYYY-MM-DD followed by an amount.\n\n"
                        f"**Extracted text sample:**\n```\n{sample}\n```"
                    ),
                    "saved_to_memory": False,
                }

            month, year = self._extract_period(transactions)
            logger.debug(
                "Observation: %d transactions | %s %d", len(transactions), month, year
            )

            # Save session summary to memory (Phase 4)
            session_summary = self._create_session_summary(transactions)
            saved = save_memory(session_summary)
            if saved:
                logger.info("Session %s %d saved to memory", month, year)
            else:
                logger.warning("Failed to save session %s %d to memory", month, year)

            return {
                "status": "success",
                "transactions": transactions,
                "count": len(transactions),
                "month": month,
                "year": year,
                "saved_to_memory": saved,
            }

        except Exception as exc:
            logger.exception("ParserAgent.run() failed")
            return {"status": "error", "error": str(exc), "saved_to_memory": False}

    def process_pdf(self, filepath: str) -> Dict[str, Any]:
        """Alias for run() — backward-compatible entry point."""
        return self.run(filepath)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_session_summary(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Build a memory-ready session summary from a transaction list.

        Includes all fields from ``calculate_monthly_summary`` plus
        placeholder ``anomaly_count`` and ``recommendations_count`` fields
        (populated by InsightsAgent on the same data).

        Args:
            transactions: Normalized transaction list.

        Returns:
            Session summary dict suitable for ``save_memory()``.
        """
        summary = calculate_monthly_summary(transactions)
        summary.setdefault("anomaly_count", 0)
        summary.setdefault("recommendations_count", 0)
        return summary

    def _extract_period(self, transactions: List[Dict]) -> Tuple[str, int]:
        """Derive statement month and year from the first transaction date (DD-Mon-YYYY)."""
        try:
            parts = transactions[0].get("date", "").split("-")
            if len(parts) == 3:
                return _MONTHS.get(parts[1], parts[1]), int(parts[2])
        except (IndexError, ValueError, AttributeError):
            pass
        return "Unknown", 0


if __name__ == "__main__":
    agent = ParserAgent()
    print(f"{agent.name} ready")
