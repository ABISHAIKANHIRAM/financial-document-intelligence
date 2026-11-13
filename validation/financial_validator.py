"""Deterministic financial validation rules and source traceability verification."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.document_loader import PageContent, ProcessedDocument
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult

logger = logging.getLogger("financial_intel.financial_validator")

STANDARD_CURRENCIES: Set[str] = {
    "EUR",
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "CNY",
    "SEK",
    "NOK",
    "DKK",
    "INR",
    "BRL",
    "SGD",
    "HKD",
}


@dataclass
class ValidationFlag:
    """An explicit deterministic validation rule flag."""

    metric_index: int
    rule_name: str
    severity: str  # "error" or "warning"
    message: str
    metric_name: str
    page_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation flag to dictionary."""
        return {
            "metric_index": self.metric_index,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "page_number": self.page_number,
        }


@dataclass
class ValidationSummary:
    """Consolidated summary of financial validation execution."""

    total_metrics: int
    verified_count: int
    flagged_count: int
    pending_count: int
    flags: List[ValidationFlag] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Return True if no metrics were flagged with errors or warnings."""
        return self.flagged_count == 0 and len(self.flags) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation summary to dictionary."""
        return {
            "total_metrics": self.total_metrics,
            "verified_count": self.verified_count,
            "flagged_count": self.flagged_count,
            "pending_count": self.pending_count,
            "flags": [f.to_dict() for f in self.flags],
            "contradictions": self.contradictions,
            "duplicates": self.duplicates,
        }


class FinancialValidator:
    """Applies deterministic financial domain rules and source evidence checks to extracted metrics."""

    def __init__(self) -> None:
        pass

    def validate_document_results(
        self,
        result: FinancialExtractionResult,
        document: Optional[ProcessedDocument] = None,
    ) -> Tuple[FinancialExtractionResult, ValidationSummary]:
        """Run all deterministic validation rules on a FinancialExtractionResult.

        Updates verification_status and flag_reasons on each metric in place.
        """
        flags: List[ValidationFlag] = []
        metrics = result.metrics

        # 1. Run metric-level deterministic checks
        for idx, m in enumerate(metrics):
            m_flags: List[ValidationFlag] = []

            # Revenue vs EBITDA confusion
            m_flags.extend(self._check_revenue_ebitda_distinction(m, idx, metrics))

            # Valid financial year
            m_flags.extend(self._check_financial_year(m, idx, result.financial_year))

            # Currency consistency
            m_flags.extend(self._check_currency_consistency(m, idx, result.currency))

            # Percentage vs absolute monetary amounts
            m_flags.extend(self._check_percentage_vs_monetary(m, idx))

            # Missing value check
            m_flags.extend(self._check_missing_value(m, idx))

            # Suspicious unit scaling
            m_flags.extend(self._check_unit_scaling(m, idx))

            # Source evidence presence & traceability
            m_flags.extend(
                self._check_source_traceability(
                    m, idx, document.pages if document else None
                )
            )

            # Assign flags to metric
            flags.extend(m_flags)
            if m_flags:
                m.verification_status = "flagged"
                m.flag_reasons = [f.message for f in m_flags]
            else:
                m.verification_status = "verified"
                m.flag_reasons = []

        # 2. Run multi-metric checks: Contradictions and Duplicates
        contradiction_flags, contradictions, duplicates = self._check_contradictions_and_duplicates(metrics)
        flags.extend(contradiction_flags)

        for c_flag in contradiction_flags:
            metric = metrics[c_flag.metric_index]
            metric.verification_status = "flagged"
            if c_flag.message not in metric.flag_reasons:
                metric.flag_reasons.append(c_flag.message)

        # Compute summary
        verified_cnt = sum(1 for m in metrics if m.verification_status == "verified")
        flagged_cnt = sum(1 for m in metrics if m.verification_status == "flagged")
        pending_cnt = sum(1 for m in metrics if m.verification_status == "pending")

        summary = ValidationSummary(
            total_metrics=len(metrics),
            verified_count=verified_cnt,
            flagged_count=flagged_cnt,
            pending_count=pending_cnt,
            flags=flags,
            contradictions=contradictions,
            duplicates=duplicates,
        )

        logger.info(
            "Validation finished for '%s': %d total, %d verified, %d flagged",
            result.document_name,
            len(metrics),
            verified_cnt,
            flagged_cnt,
        )
        return result, summary

    # --------------------------------------------------------------------------
    # Individual Rule Implementations
    # --------------------------------------------------------------------------

    def _check_revenue_ebitda_distinction(
        self,
        metric: ExtractedMetric,
        idx: int,
        all_metrics: List[ExtractedMetric],
    ) -> List[ValidationFlag]:
        """Ensure Revenue and EBITDA are not confused in naming or accounting relationships."""
        flags: List[ValidationFlag] = []
        name_lower = metric.metric.lower().strip()
        evidence_lower = metric.evidence.lower()

        # Evidence-level keyword cross-check
        if name_lower in {"revenue", "turnover", "sales"}:
            if "ebitda" in evidence_lower and not any(k in evidence_lower for k in ("revenue", "turnover", "sales")):
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="revenue_ebitda_confusion",
                        severity="error",
                        message="Metric labelled 'Revenue' but evidence text exclusively discusses EBITDA.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        if name_lower == "ebitda":
            if "ebitda" not in evidence_lower and any(k in evidence_lower for k in ("revenue increased", "total revenue", "turnover")):
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="revenue_ebitda_confusion",
                        severity="error",
                        message="Metric labelled 'EBITDA' but evidence appears to discuss top-line revenue.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        # EBITDA cannot exceed Revenue for the same company/year/period
        if name_lower == "ebitda" and metric.value is not None and metric.value > 0:
            matching_revenue = next(
                (
                    m for m in all_metrics
                    if m.metric.lower().strip() in {"revenue", "turnover", "sales"}
                    and m.year == metric.year
                    and m.value is not None
                    and m.value > 0
                ),
                None,
            )
            if matching_revenue and metric.value > matching_revenue.value:
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="ebitda_exceeds_revenue",
                        severity="error",
                        message=(
                            f"EBITDA ({metric.value:,.0f}) exceeds reported Revenue ({matching_revenue.value:,.0f}) "
                            f"for fiscal year {metric.year}."
                        ),
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        return flags

    def _check_financial_year(
        self,
        metric: ExtractedMetric,
        idx: int,
        document_year: Optional[int],
    ) -> List[ValidationFlag]:
        """Validate fiscal year validity and document consistency."""
        flags: List[ValidationFlag] = []

        if metric.year is None:
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="missing_financial_year",
                    severity="warning",
                    message="Metric lacks an explicit financial year.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )
            return flags

        # Realistic year range
        if not (1980 <= metric.year <= 2040):
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="invalid_financial_year",
                    severity="error",
                    message=f"Financial year {metric.year} is outside standard operating range (1980-2040).",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )

        # Discrepancy with document primary year
        if document_year and abs(metric.year - document_year) > 4:
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="year_document_mismatch",
                    severity="warning",
                    message=f"Metric year ({metric.year}) diverges significantly from document primary year ({document_year}).",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )

        return flags

    def _check_currency_consistency(
        self,
        metric: ExtractedMetric,
        idx: int,
        document_currency: Optional[str],
    ) -> List[ValidationFlag]:
        """Validate currency code validity and document-level consistency."""
        flags: List[ValidationFlag] = []

        # If metric is percentage or headcount, currency check does not apply
        if metric.unit in {"%", "count", "employees"}:
            return flags

        if not metric.unit:
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="missing_currency",
                    severity="warning",
                    message="Monetary metric is missing currency code.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )
            return flags

        unit_clean = metric.unit.strip().upper()
        if unit_clean not in STANDARD_CURRENCIES:
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="unrecognized_currency",
                    severity="warning",
                    message=f"Currency unit '{metric.unit}' is not in standard ISO currency table.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )

        if document_currency and unit_clean != document_currency.strip().upper():
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="currency_mismatch",
                    severity="warning",
                    message=f"Metric currency '{metric.unit}' differs from document currency '{document_currency}'.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )

        return flags

    def _check_percentage_vs_monetary(
        self,
        metric: ExtractedMetric,
        idx: int,
    ) -> List[ValidationFlag]:
        """Distinguish percentages from absolute monetary amounts."""
        flags: List[ValidationFlag] = []
        name_lower = metric.metric.lower().strip()

        is_percentage_metric = any(term in name_lower for term in ("growth", "margin", "percentage", "rate"))
        is_absolute_metric = any(
            name_lower == term
            for term in (
                "revenue",
                "ebitda",
                "ebit",
                "net income",
                "total assets",
                "total liabilities",
                "cash flow",
            )
        )

        if is_percentage_metric:
            if metric.unit != "%":
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="percentage_unit_mismatch",
                        severity="error",
                        message=f"Metric '{metric.metric}' represents a rate/percentage but has unit '{metric.unit}'.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )
            if metric.value is not None and abs(metric.value) > 1000.0:
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="suspicious_percentage_value",
                        severity="error",
                        message=f"Percentage value {metric.value}% is abnormally large.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        if is_absolute_metric:
            if metric.unit == "%":
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="absolute_metric_has_percentage",
                        severity="error",
                        message=f"Absolute financial metric '{metric.metric}' cannot have percentage unit '%'.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        return flags

    def _check_missing_value(
        self,
        metric: ExtractedMetric,
        idx: int,
    ) -> List[ValidationFlag]:
        """Ensure missing values are preserved as null and flagged for review."""
        flags: List[ValidationFlag] = []
        if metric.value is None:
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="null_value_requires_review",
                    severity="warning",
                    message="Metric value is null; requires manual confirmation or document verification.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )
        return flags

    def _check_unit_scaling(
        self,
        metric: ExtractedMetric,
        idx: int,
    ) -> List[ValidationFlag]:
        """Detect unscaled numbers where evidence specifies millions or thousands."""
        flags: List[ValidationFlag] = []
        if metric.value is None or metric.unit == "%":
            return flags

        evidence_lower = metric.evidence.lower()
        val_abs = abs(metric.value)

        # Check for unscaled millions
        if re.search(r"\b(?:million|mn|mio|m)\b", evidence_lower):
            if 0.0 < val_abs < 1000.0:
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="suspicious_unscaled_value",
                        severity="warning",
                        message=(
                            f"Evidence text mentions 'million', but extracted value ({metric.value}) "
                            "appears to be unscaled."
                        ),
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        # Check for unscaled thousands
        if re.search(r"\b(?:thousand|k|tsd)\b", evidence_lower):
            if 0.0 < val_abs < 500.0:
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="suspicious_unscaled_value",
                        severity="warning",
                        message=(
                            f"Evidence text mentions 'thousand', but extracted value ({metric.value}) "
                            "appears to be unscaled."
                        ),
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )

        return flags

    def _check_source_traceability(
        self,
        metric: ExtractedMetric,
        idx: int,
        document_pages: Optional[List[PageContent]],
    ) -> List[ValidationFlag]:
        """Verify source evidence existence and trace to document page."""
        flags: List[ValidationFlag] = []

        if not metric.evidence or not metric.evidence.strip():
            flags.append(
                ValidationFlag(
                    metric_index=idx,
                    rule_name="missing_source_evidence",
                    severity="error",
                    message="Metric lacks source evidence text.",
                    metric_name=metric.metric,
                    page_number=metric.source.page,
                )
            )
            return flags

        # If document pages are available, verify evidence against page text
        if document_pages:
            target_page = next(
                (p for p in document_pages if p.page_number == metric.source.page),
                None,
            )
            if not target_page:
                flags.append(
                    ValidationFlag(
                        metric_index=idx,
                        rule_name="invalid_source_page",
                        severity="error",
                        message=f"Cited source page {metric.source.page} does not exist in the document.",
                        metric_name=metric.metric,
                        page_number=metric.source.page,
                    )
                )
            else:
                # Substring check with normalized whitespace
                clean_page_text = re.sub(r"\s+", " ", target_page.text).lower()
                clean_evidence = re.sub(r"\s+", " ", metric.evidence).lower().strip()

                # Check if entire evidence or primary snippet appears in page
                evidence_words = clean_evidence.split()
                snippet = " ".join(evidence_words[:6]) if len(evidence_words) >= 6 else clean_evidence

                if snippet not in clean_page_text:
                    flags.append(
                        ValidationFlag(
                            metric_index=idx,
                            rule_name="unverified_source_evidence",
                            severity="warning",
                            message=f"Evidence text could not be verified on page {metric.source.page}.",
                            metric_name=metric.metric,
                            page_number=metric.source.page,
                        )
                    )

        return flags

    def _check_contradictions_and_duplicates(
        self,
        metrics: List[ExtractedMetric],
    ) -> Tuple[List[ValidationFlag], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect identical duplicates and conflicting values across different pages."""
        flags: List[ValidationFlag] = []
        contradictions: List[Dict[str, Any]] = []
        duplicates: List[Dict[str, Any]] = []

        # Group by (normalized_metric, year, unit)
        groups: Dict[Tuple[str, Optional[int], Optional[str]], List[Tuple[int, ExtractedMetric]]] = {}
        for idx, m in enumerate(metrics):
            key = (m.metric.lower().strip(), m.year, m.unit)
            groups.setdefault(key, []).append((idx, m))

        for key, group_metrics in groups.items():
            if len(group_metrics) <= 1:
                continue

            metric_name, year, unit = key
            unique_values = {m.value for _, m in group_metrics if m.value is not None}

            # Contradiction: same metric, same year, DIFFERENT numeric values
            if len(unique_values) > 1:
                details = [
                    {"page": m.source.page, "value": m.value, "evidence": m.evidence}
                    for _, m in group_metrics
                ]
                contradictions.append(
                    {
                        "metric": metric_name,
                        "year": year,
                        "unit": unit,
                        "occurrences": details,
                    }
                )
                for idx, m in group_metrics:
                    flags.append(
                        ValidationFlag(
                            metric_index=idx,
                            rule_name="contradictory_metric_values",
                            severity="error",
                            message=(
                                f"Contradictory values detected for {m.metric} ({year}): "
                                f"Page {m.source.page} reports {m.value}, differing from other pages."
                            ),
                            metric_name=m.metric,
                            page_number=m.source.page,
                        )
                    )

            # Duplicate: same metric, same year, SAME numeric value from multiple pages
            elif len(unique_values) == 1:
                pages = [m.source.page for _, m in group_metrics]
                duplicates.append(
                    {
                        "metric": metric_name,
                        "year": year,
                        "value": next(iter(unique_values)),
                        "pages": pages,
                    }
                )

        return flags, contradictions, duplicates
