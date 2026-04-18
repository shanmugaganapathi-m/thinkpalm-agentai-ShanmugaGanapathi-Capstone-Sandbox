"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_pdf_path():
    """Path to the real sample PDF (used in later phases)."""
    return Path("sample_data/Shanmuga_Mar2026.pdf")


@pytest.fixture
def sample_pdf(tmp_path):
    """
    A real, minimal PDF containing bank-statement-style rows.

    Created programmatically so tests never depend on a checked-in binary.
    """
    import fitz  # PyMuPDF

    pdf_path = tmp_path / "test_statement.pdf"
    doc = fitz.open()
    page = doc.new_page()

    rows = [
        "Bank Statement – March 2026",
        "",
        "Date        Description                        Debit       Credit      Balance",
        "02-Mar-2026 Salary Credit - Mirae Asset                   132000.00   132000.00",
        "03-Mar-2026 SWIGGY FOOD ORDER                 450.00                  131550.00",
        "05-Mar-2026 HDFC HOME LOAN EMI                28450.00                103100.00",
        "10-Mar-2026 NETFLIX SUBSCRIPTION              649.00                  102451.00",
        "12-Mar-2026 AMAZON SHOPPING                   1299.00                 101152.00",
        "15-Mar-2026 BESCOM ELECTRICITY BILL           1200.00                  99952.00",
    ]

    y = 50
    for row in rows:
        page.insert_text((30, y), row, fontsize=9)
        y += 14

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_transactions():
    """Sample normalized transactions for testing."""
    return [
        {
            "date": "02-Mar-2026",
            "merchant": "Salary",
            "description": "Salary Credit - Mirae Asset",
            "amount": 132000.0,
            "type": "income",
            "category": "Income",
        },
        {
            "date": "03-Mar-2026",
            "merchant": "Swiggy",
            "description": "SWIGGY FOOD ORDER",
            "amount": 450.0,
            "type": "expense",
            "category": "Food",
        },
        {
            "date": "05-Mar-2026",
            "merchant": "HDFC",
            "description": "HDFC HOME LOAN EMI",
            "amount": 28450.0,
            "type": "expense",
            "category": "EMI/Loans",
        },
    ]


@pytest.fixture
def sample_summary():
    """Sample monthly summary for testing."""
    return {
        "month": "March",
        "year": 2026,
        "total_income": 132000.0,
        "total_spend": 103038.0,
        "net_savings": 28962.0,
        "savings_rate": 21.95,
        "categories": {
            "Food": 2070.0,
            "Groceries": 8340.0,
            "EMI/Loans": 28450.0,
            "Insurance": 8500.0,
            "Utilities": 1200.0,
            "Entertainment": 398.0,
            "Shopping": 10498.0,
            "Transport": 620.0,
            "Health": 920.0,
            "Savings/Investment": 15000.0,
            "Education": 4500.0,
            "Other": 5025.0,
        },
        "transaction_count": 31,
    }


@pytest.fixture
def temp_memory_file(tmp_path):
    """Temporary memory file path for memory-manager tests."""
    return tmp_path / "user_memory.json"
