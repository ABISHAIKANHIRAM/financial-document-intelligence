"""Interactive financial extraction results viewer, source evidence inspector, and exporter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from exports.export_manager import ExportManager
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult
from validation.financial_validator import ValidationSummary


def render_results() -> None:
    """Render the extracted financial results, source evidence inspector, and export panel."""
    st.header("Financial Extraction Results")
    st.caption("Inspect extracted financial statements, audit source evidence, and download structured outputs.")

    result: FinancialExtractionResult = st.session_state.get("active_result")
    doc = st.session_state.get("active_document")
    summary: ValidationSummary = st.session_state.get("active_summary")

    if not result:
        st.info("No active extraction result loaded. Please upload a document or select one from Document History.")
        return

    # 1. Company Metadata Header Card
    with st.container():
        st.subheader(f"🏢 {result.company_name or 'Company Overview'}")
        col_meta1, col_meta2, col_meta3, col_meta4, col_meta5 = st.columns(5)
        with col_meta1:
            st.markdown(f"**Fiscal Year:** `{result.financial_year or 'N/A'}`")
        with col_meta2:
            st.markdown(f"**Reporting Period:** `{result.reporting_period or 'N/A'}`")
        with col_meta3:
            st.markdown(f"**Reporting Currency:** `{result.currency or 'N/A'}`")
        with col_meta4:
            st.markdown(f"**Country:** `{result.country or 'N/A'}`")
        with col_meta5:
            st.markdown(f"**Industry:** `{result.industry or 'N/A'}`")

        st.caption(
            f"📄 Source: **{result.document_name}** | 🤖 Local LLM: `{result.model_used or 'Default'}` | ⏰ Processed: {result.created_at[:19]}"
        )

    st.divider()

    # Tabs for organization
    tab_metrics, tab_traceability, tab_audit, tab_export = st.tabs(
        [
            "📊 Structured Financial Metrics",
            "🔍 Source Traceability Inspector",
            "🛡️ Validation & Quality Audit",
            "💾 JSON Viewer & Exports",
        ]
    )

    # Tab 1: Structured Metrics
    with tab_metrics:
        st.subheader("Extracted Financial Metrics")

        # Metric filters
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            search_query = st.text_input("Filter metrics by name", placeholder="e.g. Revenue, EBITDA, Cash Flow")
        with f_col2:
            status_filter = st.selectbox(
                "Filter by Verification Status",
                options=["All", "Verified", "Flagged (Review Required)", "Pending"],
            )

        filtered_metrics = result.metrics
        if search_query:
            filtered_metrics = [
                m for m in filtered_metrics if search_query.lower() in m.metric.lower()
            ]
        if status_filter == "Verified":
            filtered_metrics = [m for m in filtered_metrics if m.verification_status == "verified"]
        elif status_filter == "Flagged (Review Required)":
            filtered_metrics = [m for m in filtered_metrics if m.verification_status == "flagged"]
        elif status_filter == "Pending":
            filtered_metrics = [m for m in filtered_metrics if m.verification_status == "pending"]

        if filtered_metrics:
            rows = []
            for m in filtered_metrics:
                # Format numeric value nicely
                val_str = f"{m.value:,.2f}" if m.value is not None else "null"
                status_badge = (
                    "🟢 Verified"
                    if m.verification_status == "verified"
                    else ("🔴 Flagged" if m.verification_status == "flagged" else "🔵 Pending")
                )
                rows.append(
                    {
                        "Metric": m.metric,
                        "Value": val_str,
                        "Unit": m.unit or "-",
                        "Year": m.year or "-",
                        "Period": m.reporting_period or "-",
                        "Status": status_badge,
                        "Page": f"p. {m.source.page}",
                        "Evidence": m.evidence,
                        "Issues / Flags": "; ".join(m.flag_reasons) if m.flag_reasons else "-",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No metrics match the selected filter criteria.")

    # Tab 2: Source Traceability Inspector
    with tab_traceability:
        st.subheader("Source Traceability & Evidence Inspector")
        st.write(
            "Verify every financial value against its original document page and verbatim text evidence."
        )

        if not result.metrics:
            st.info("No extracted metrics available.")
        else:
            metric_options = [
                f"{m.metric} ({m.year or 'N/A'}) - Page {m.source.page}: {m.value} {m.unit or ''}"
                for m in result.metrics
            ]
            selected_idx = st.selectbox(
                "Select a metric to inspect source provenance:",
                range(len(metric_options)),
                format_func=lambda idx: metric_options[idx],
            )
            selected_metric: ExtractedMetric = result.metrics[selected_idx]

            col_ev1, col_ev2 = st.columns([1, 1])

            with col_ev1:
                st.markdown("#### Extracted Metric Provenance")
                st.info(
                    f"**Metric:** `{selected_metric.metric}`\n\n"
                    f"**Value:** `{selected_metric.value}` {selected_metric.unit or ''}\n\n"
                    f"**Fiscal Year:** `{selected_metric.year or 'N/A'}`\n\n"
                    f"**Document:** `{selected_metric.source.document}`\n\n"
                    f"**Page Number:** `Page {selected_metric.source.page}`\n\n"
                    f"**Extraction Method:** `{selected_metric.extraction_method}`\n\n"
                    f"**Status:** `{selected_metric.verification_status.upper()}`"
                )
                st.markdown("#### Verbatim Evidence Quote:")
                st.success(f"> \"{selected_metric.evidence}\"")

                if selected_metric.flag_reasons:
                    st.warning("**Validation Review Flags:**\n" + "\n".join(f"- {r}" for r in selected_metric.flag_reasons))

            with col_ev2:
                st.markdown(f"#### Original Page {selected_metric.source.page} Content")
                target_page = None
                if doc:
                    target_page = doc.get_page(selected_metric.source.page)

                if target_page and target_page.text:
                    st.text_area(
                        f"Page {selected_metric.source.page} Text Layer",
                        value=target_page.text,
                        height=350,
                        disabled=True,
                    )
                else:
                    st.caption("Raw page text preview not loaded in current session.")

    # Tab 3: Validation & Quality Audit
    with tab_audit:
        st.subheader("Validation Rules & Accounting Audit")
        if summary:
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Total Flags", len(summary.flags))
            with col_a2:
                st.metric("Contradictions Found", len(summary.contradictions))
            with col_a3:
                st.metric("Duplicate Metric Groups", len(summary.duplicates))

            if summary.contradictions:
                st.error("⚠️ **Contradictory Values Detected Across Pages:**")
                for c in summary.contradictions:
                    st.write(f"- **{c['metric'].title()} ({c['year']})**: Conflicting values across pages: {c['occurrences']}")

            if summary.duplicates:
                st.info("ℹ️ **Duplicate Metrics Detected (Repeated across pages):**")
                for d in summary.duplicates:
                    st.write(f"- **{d['metric'].title()} ({d['year']})**: Value `{d['value']}` on pages {d['pages']}")

            if summary.flags:
                st.markdown("#### Detailed Rule Violations & Flags:")
                flag_rows = [
                    {
                        "Rule": f.rule_name,
                        "Metric": f.metric_name,
                        "Page": f.page_number,
                        "Severity": f.severity.upper(),
                        "Message": f.message,
                    }
                    for f in summary.flags
                ]
                st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
            else:
                st.success("🎉 All extracted metrics passed deterministic validation rules with zero errors.")
        else:
            st.info("Validation summary not available.")

    # Tab 4: JSON Viewer & Exports
    with tab_export:
        st.subheader("JSON Viewer & Formatted Exports")
        st.caption("Download structured data in JSON, CSV, or multi-sheet Excel format.")

        # Action buttons for download
        json_bytes = ExportManager.to_json_bytes(result)
        csv_bytes = ExportManager.to_csv_bytes(result)
        excel_bytes = ExportManager.to_excel_bytes(result, validation_summary=summary)

        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            st.download_button(
                label="📥 Download Structured JSON",
                data=json_bytes,
                file_name=f"{Path(result.document_name).stem}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )
        with btn_c2:
            st.download_button(
                label="📥 Export CSV Metrics",
                data=csv_bytes,
                file_name=f"{Path(result.document_name).stem}_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with btn_c3:
            st.download_button(
                label="📥 Export Excel Workbook (.xlsx)",
                data=excel_bytes,
                file_name=f"{Path(result.document_name).stem}_financial_intel.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### Formatted Pydantic JSON Payload:")
        st.json(result.model_dump())
