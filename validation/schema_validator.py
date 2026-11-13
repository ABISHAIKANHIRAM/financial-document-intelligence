"""Pydantic schema validation and structural constraint enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from schemas.financial_schema import (
    ChunkExtractionResponse,
    ExtractedMetric,
    FinancialExtractionResult,
    SourceReference,
)

logger = logging.getLogger("financial_intel.schema_validator")


@dataclass
class ValidationErrorItem:
    """Individual field validation error."""

    field_path: str
    message: str
    invalid_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation error item to dictionary."""
        return {
            "field_path": self.field_path,
            "message": self.message,
            "invalid_value": self.invalid_value,
        }


@dataclass
class SchemaValidationReport:
    """Outcome of structural schema validation."""

    is_valid: bool
    errors: List[ValidationErrorItem] = field(default_factory=list)
    validated_model: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "error_count": len(self.errors),
        }


class SchemaValidator:
    """Validates raw and structured financial data against Pydantic models and field boundaries."""

    VALID_METRIC_NAMES = {
        "revenue",
        "turnover",
        "sales",
        "ebitda",
        "ebit",
        "operating profit",
        "operating income",
        "net income",
        "net profit",
        "net loss",
        "total assets",
        "total liabilities",
        "cash flow",
        "operating cash flow",
        "free cash flow",
        "number of employees",
        "employees",
        "headcount",
        "growth percentage",
        "revenue growth",
    }

    def validate_metric(self, metric_input: Dict[str, Any] | ExtractedMetric) -> SchemaValidationReport:
        """Validate an individual ExtractedMetric or raw dictionary against schema rules."""
        errors: List[ValidationErrorItem] = []
        validated: Optional[ExtractedMetric] = None

        if isinstance(metric_input, ExtractedMetric):
            validated = metric_input
        elif isinstance(metric_input, dict):
            try:
                validated = ExtractedMetric.model_validate(metric_input)
            except ValidationError as val_err:
                for err in val_err.errors():
                    loc = " -> ".join(str(p) for p in err["loc"])
                    errors.append(
                        ValidationErrorItem(
                            field_path=loc,
                            message=err["msg"],
                            invalid_value=err.get("input"),
                        )
                    )
                return SchemaValidationReport(is_valid=False, errors=errors)
        else:
            errors.append(
                ValidationErrorItem(
                    field_path="metric",
                    message=f"Unsupported input type: {type(metric_input).__name__}",
                )
            )
            return SchemaValidationReport(is_valid=False, errors=errors)

        # Domain boundary checks on validated metric
        if validated.year is not None and not (1900 <= validated.year <= 2100):
            errors.append(
                ValidationErrorItem(
                    field_path="year",
                    message=f"Financial year {validated.year} is out of realistic range (1900-2100).",
                    invalid_value=validated.year,
                )
            )

        metric_lower = validated.metric.lower().strip()
        if metric_lower in {"number of employees", "employees", "headcount"}:
            if validated.value is not None and validated.value < 0:
                errors.append(
                    ValidationErrorItem(
                        field_path="value",
                        message=f"Employee headcount cannot be negative: {validated.value}",
                        invalid_value=validated.value,
                    )
                )

        if not validated.evidence or len(validated.evidence.strip()) < 3:
            errors.append(
                ValidationErrorItem(
                    field_path="evidence",
                    message="Verbatim source evidence cannot be empty or trivially short.",
                    invalid_value=validated.evidence,
                )
            )

        if validated.source.page < 1:
            errors.append(
                ValidationErrorItem(
                    field_path="source.page",
                    message=f"Page number must be >= 1, got {validated.source.page}",
                    invalid_value=validated.source.page,
                )
            )

        is_valid = len(errors) == 0
        return SchemaValidationReport(
            is_valid=is_valid,
            errors=errors,
            validated_model=validated if is_valid else None,
        )

    def validate_extraction_result(
        self, result_input: Dict[str, Any] | FinancialExtractionResult
    ) -> SchemaValidationReport:
        """Validate a full document-level extraction result."""
        errors: List[ValidationErrorItem] = []
        validated: Optional[FinancialExtractionResult] = None

        if isinstance(result_input, FinancialExtractionResult):
            validated = result_input
        elif isinstance(result_input, dict):
            try:
                validated = FinancialExtractionResult.model_validate(result_input)
            except ValidationError as val_err:
                for err in val_err.errors():
                    loc = " -> ".join(str(p) for p in err["loc"])
                    errors.append(
                        ValidationErrorItem(
                            field_path=loc,
                            message=err["msg"],
                            invalid_value=err.get("input"),
                        )
                    )
                return SchemaValidationReport(is_valid=False, errors=errors)
        else:
            return SchemaValidationReport(
                is_valid=False,
                errors=[
                    ValidationErrorItem(
                        field_path="result",
                        message=f"Expected FinancialExtractionResult or dict, got {type(result_input).__name__}",
                    )
                ],
            )

        # Validate each individual metric
        for idx, m in enumerate(validated.metrics):
            m_report = self.validate_metric(m)
            for err in m_report.errors:
                errors.append(
                    ValidationErrorItem(
                        field_path=f"metrics[{idx}].{err.field_path}",
                        message=err.message,
                        invalid_value=err.invalid_value,
                    )
                )

        is_valid = len(errors) == 0
        return SchemaValidationReport(
            is_valid=is_valid,
            errors=errors,
            validated_model=validated if is_valid else None,
        )
