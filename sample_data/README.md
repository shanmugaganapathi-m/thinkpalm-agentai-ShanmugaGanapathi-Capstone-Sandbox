# Sample Data

This directory should contain sample bank statements for testing.

## Adding Sample Data

1. Export your bank statement as PDF (e.g., IndusInd Bank)
2. Place the PDF file here: `Shanmuga_Mar2026.pdf`
3. Run tests:
   ```bash
   pytest tests/test_parser_agent.py -v
   ```

## File Format

Supported format: **PDF** (bank statement format)

Example: `Shanmuga_Mar2026.pdf`

## Privacy

- Do not commit actual bank statements to the repo
- Add `sample_data/*.pdf` to `.gitignore` if needed
- Use sanitized/anonymized sample statements for testing

---

To be added in Phase 2.
