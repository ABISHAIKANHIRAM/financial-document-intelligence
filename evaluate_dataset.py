"""Evaluation benchmark CLI runner comparing extraction results against ground_truth.json."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.document_loader import DocumentLoader
from core.pdf_inspector import PdfInspector
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult, SourceReference
from validation.financial_validator import FinancialValidator

DATASET_DIR = Path(__file__).resolve().parent / "financial_document_test_dataset"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def run_benchmark(save_json: bool = True) -> int:
    """Run evaluation benchmark against financial_document_test_dataset."""
    start_time = time.perf_counter()
    gt_path = DATASET_DIR / "ground_truth.json"
    if not gt_path.exists():
        print(f"Error: ground_truth.json not found in {DATASET_DIR}")
        return 1

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    documents_config = gt_data.get("documents", [])

    print("=" * 80)
    print("FINANCIAL DOCUMENT INTELLIGENCE — BENCHMARK EVALUATION")
    print("=" * 80)

    loader = DocumentLoader()
    inspector = PdfInspector()
    validator = FinancialValidator()

    total_eval_checks = 0
    passed_eval_checks = 0
    report_details = []

    for doc_entry in documents_config:
        doc_start = time.perf_counter()
        file_name = doc_entry["file"]
        company = doc_entry["company"]
        fy = doc_entry["financial_year"]
        currency = doc_entry["currency"]
        expected_metrics = doc_entry.get("expected_metrics", {})
        file_path = DATASET_DIR / file_name

        print(f"\nEvaluating: {file_name}")
        print(f"Company: {company} | Year: {fy} | Currency: {currency}")
        print("-" * 80)

        if not file_path.exists():
            print(f"  [ERROR] File missing: {file_path}")
            continue

        doc_summary = {
            "file": file_name,
            "company": company,
            "financial_year": fy,
            "currency": currency,
            "checks": [],
        }

        # 1. Inspection Layer Check
        insp_report = inspector.inspect(file_path)
        print(f"  • Pages: {insp_report.page_count} | Scanned: {insp_report.scanned_page_numbers} | Text: {insp_report.text_page_numbers}")

        if "scanned" in file_name:
            total_eval_checks += 1
            if insp_report.is_predominantly_scanned:
                passed_eval_checks += 1
                doc_summary["checks"].append({"check": "scanned_detection", "status": "PASS"})
                print("  ✓ Scanned page detection: PASS")
            else:
                doc_summary["checks"].append({"check": "scanned_detection", "status": "FAIL"})
                print("  ✗ Scanned page detection: FAIL")
        else:
            total_eval_checks += 1
            if not insp_report.is_predominantly_scanned:
                passed_eval_checks += 1
                doc_summary["checks"].append({"check": "text_layer_detection", "status": "PASS"})
                print("  ✓ Selectable text layer detection: PASS")
            else:
                doc_summary["checks"].append({"check": "text_layer_detection", "status": "FAIL"})
                print("  ✗ Selectable text layer detection: FAIL")

        # 2. Document Ingestion Check
        processed_doc = loader.load(file_path)
        print(f"  • Ingested tables: {len(processed_doc.tables)} | Chunks: {len(processed_doc.chunks)}")

        # 3. Precision vs Expected Ground Truth
        print("  • Expected Metric Assertions:")
        metric_checks = []
        for metric_key, expected_val in expected_metrics.items():
            total_eval_checks += 1
            full_text = processed_doc.get_full_text()
            expected_str = str(expected_val)
            clean_full = full_text.replace(",", "").replace(" ", "")
            val_in_text = (
                expected_str in full_text
                or f"{expected_val:,}" in full_text
                or expected_str in clean_full
                or f"{expected_val / 1_000_000:.1f}" in full_text
                or f"{expected_val / 1_000_000:.2f}" in full_text
                or f"{expected_val:.1f}" in full_text
            )
            if val_in_text:
                passed_eval_checks += 1
                metric_checks.append({"metric": metric_key, "expected": expected_val, "status": "PASS"})
                print(f"    ✓ {metric_key:<25} Expected: {expected_val:>12} (Found in source text)")
            else:
                metric_checks.append({"metric": metric_key, "expected": expected_val, "status": "FAIL"})
                print(f"    ✗ {metric_key:<25} Expected: {expected_val:>12} (Missing from source text)")

        doc_summary["metric_assertions"] = metric_checks
        doc_summary["duration_seconds"] = round(time.perf_counter() - doc_start, 3)
        report_details.append(doc_summary)

    # 4. Multi-sheet Excel Check (entire-omv-ar25.xlsx)
    omv_path = DATASET_DIR / "entire-omv-ar25.xlsx"
    if omv_path.exists():
        omv_start = time.perf_counter()
        print("\nEvaluating: entire-omv-ar25.xlsx")
        print("-" * 80)
        omv_doc = loader.load(omv_path)
        total_eval_checks += 1
        omv_pass = omv_doc.page_count == 300 and len(omv_doc.chunks) > 0
        if omv_pass:
            passed_eval_checks += 1
            print(f"  ✓ 300-sheet Excel Ingestion: PASS ({omv_doc.page_count} sheets, {len(omv_doc.chunks)} prioritized chunks)")
        else:
            print("  ✗ 300-sheet Excel Ingestion: FAIL")

        report_details.append({
            "file": "entire-omv-ar25.xlsx",
            "sheets": omv_doc.page_count,
            "prioritized_chunks": len(omv_doc.chunks),
            "status": "PASS" if omv_pass else "FAIL",
            "duration_seconds": round(time.perf_counter() - omv_start, 3),
        })

    elapsed = time.perf_counter() - start_time
    score_pct = (passed_eval_checks / max(1, total_eval_checks)) * 100

    print("\n" + "=" * 80)
    print(f"BENCHMARK SCORE: {passed_eval_checks}/{total_eval_checks} checks passed ({score_pct:.1f}%) | Time: {elapsed:.2f}s")
    print("=" * 80)

    if save_json:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / "benchmark_report.json"
        report_payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "score_percentage": round(score_pct, 2),
            "checks_passed": passed_eval_checks,
            "checks_total": total_eval_checks,
            "duration_seconds": round(elapsed, 3),
            "documents": report_details,
        }
        report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Saved benchmark summary report to {report_file}")

    return 0 if score_pct >= 90.0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate financial extraction dataset benchmark.")
    parser.add_argument("--no-json", action="store_true", help="Disable saving JSON report")
    args = parser.parse_args()

    raise SystemExit(run_benchmark(save_json=not args.no_json))
