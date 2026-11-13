# Financial Document Intelligence — Test Dataset

All companies, figures, and documents are fictional and synthetic. Do not treat these as real financial disclosures.

## Files
1. `01_northstar_annual_report_2025.pdf` — text-based PDF with a financial table and narrative.
2. `02_northstar_press_release_2025.pdf` — text-based press release with preliminary values that intentionally differ from the final report.
3. `03_alpine_works_scanned_statement.pdf` — image-only PDF to test OCR fallback.
4. `ground_truth.json` — expected values for evaluation.

## Suggested tests
- Check whether the application detects a text layer in files 1 and 2.
- Check whether it detects file 3 as scanned/image-only and invokes OCR.
- Compare extracted metrics with `ground_truth.json`.
- Ensure it distinguishes average employees (81) from year-end employees (84).
- Ensure preliminary revenue/EBITDA in the press release are not silently merged with final annual-report figures.
- Ensure every extracted value has a real document/page/evidence reference.
- Missing values must remain null rather than being guessed.

## Evaluation
For each document, compare extracted fields to ground truth. Report:
- Exact numeric match rate
- Correct metric classification
- Correct year and currency
- Evidence/source coverage
- JSON schema validity
- OCR success for the scanned PDF

Use tolerances only where your extraction design explicitly normalizes rounding or units.
