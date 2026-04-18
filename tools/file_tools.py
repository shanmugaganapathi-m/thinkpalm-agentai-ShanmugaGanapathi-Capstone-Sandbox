"""File I/O tools for PDF reading and validation."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def read_pdf(filepath: str) -> Optional[str]:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        Extracted text as a single string, or None on any error.

    Example:
        >>> text = read_pdf("sample_data/statement.pdf")
        >>> print(text[:80])
        'Date        Description          Debit    Credit\\n02-Mar-2026 Salary ...'
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")
        return None

    try:
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return None
        if path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {filepath}")
            return None

        doc = fitz.open(str(path))
        pages_text: list[str] = [page.get_text() for page in doc]
        doc.close()

        text = "\n".join(pages_text)
        logger.debug(f"Extracted {len(text)} characters from {filepath}")
        return text

    except Exception as exc:
        logger.error(f"Error reading PDF '{filepath}': {exc}")
        return None


def validate_pdf(filepath: str) -> bool:
    """
    Validate that a file exists, has a .pdf extension, and can be opened.

    Args:
        filepath: Path to the file to validate.

    Returns:
        True if the file is a valid, openable PDF; False otherwise.

    Example:
        >>> validate_pdf("sample_data/statement.pdf")
        True
        >>> validate_pdf("missing.txt")
        False
    """
    try:
        path = Path(filepath)

        if not path.exists():
            logger.error(f"Validation failed – file not found: {filepath}")
            return False

        if path.suffix.lower() != ".pdf":
            logger.error(f"Validation failed – not a .pdf extension: {filepath}")
            return False

        try:
            import fitz
            doc = fitz.open(str(path))
            doc.close()
        except Exception as exc:
            logger.error(f"Validation failed – cannot open PDF '{filepath}': {exc}")
            return False

        logger.debug(f"PDF validation passed: {filepath}")
        return True

    except Exception as exc:
        logger.error(f"PDF validation error for '{filepath}': {exc}")
        return False
