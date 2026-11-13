"""Document history, audit trail, cross-document analytics, and database maintenance view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from database.repository import DocumentRepository
from schemas.financial_schema import FinancialExtractionResult
from validation.financial_validator import ValidationSummary


def render_history(repo: DocumentRepository) -> None:
    """Render the document archive, audit trail, and cross-document analytics view."""
    st.header("Document Archive & History")
    st.caption("Access historical extractions, audit trail events, cross-document trend analytics, and database maintenance.")

    tab_docs, tab_analytics, tab_logs, tab_db = st.tabs(
        [
            "🗂️ Stored Documents",
            "📈 Cross-Document Trends",
            "📜 Audit Event Logs",
            "💾 Database Maintenance",
        ]
    )

    # --------------------------------------------------------------------------
    # Tab 1: Stored Documents with Search & Filters
    # --------------------------------------------------------------------------
    with tab_docs:
        st.subheader("Archived Extraction Records")

        col_search1, col_search2, col_search3 = st.columns([2, 1, 1])
        with col_search1:
            query = st.text_input("Search by company or filename", placeholder="e.g. Northstar, OMV, annual_report")
        with col_search2:
            status_filter = st.selectbox(
                "Filter by Quality",
                options=["All Records", "Verified Only", "Needs Review (Flagged)"],
            )
        with col_search3:
            year_val = st.text_input("Filter by Year", placeholder="e.g. 2025")

        filter_code = None
        if status_filter == "Verified Only":
            filter_code = "verified_only"
        elif status_filter == "Needs Review (Flagged)":
            filter_code = "has_flags"

        year_int = None
        if year_val and year_val.strip().isdigit():
            year_int = int(year_val.strip())

        documents = repo.search_documents(
            query=query,
            year=year_int,
            status_filter=filter_code,
            limit=100,
        )

        if not documents:
            st.info("No matching documents found in archive.")
        else:
            st.caption(f"Showing **{len(documents)}** records:")

            for doc in documents:
                with st.expander(
                    f"📄 #{doc['id']}: {doc['filename']} — {doc['company_name'] or 'Unknown Company'} ({doc['financial_year'] or 'Year N/A'})"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.write(f"**Format:** {doc['file_type'].upper()}")
                        st.write(f"**Pages/Sheets:** {doc['page_count']}")
                    with c2:
                        st.write(f"**Total Metrics:** {doc['metric_count']}")
                        st.write(f"**Verified:** 🟢 {doc['verified_count'] or 0}")
                    with c3:
                        st.write(f"**Flagged:** 🔴 {doc['flagged_count'] or 0}")
                        st.write(f"**Currency:** {doc['currency'] or 'N/A'}")
                    with c4:
                        st.write(f"**Model:** `{doc['model_used'] or 'Default'}`")
                        st.write(f"**Processed:** {doc['created_at'][:19].replace('T', ' ')}")

                    btn_col1, btn_col2 = st.columns([1, 1])
                    with btn_col1:
                        if st.button(
                            "🔍 Load into Results & Export Viewer",
                            key=f"load_doc_{doc['id']}",
                            use_container_width=True,
                        ):
                            full_record = repo.get_document(doc["id"])
                            if full_record and full_record.get("extraction_data"):
                                try:
                                    res_model = FinancialExtractionResult.model_validate(
                                        full_record["extraction_data"]
                                    )
                                    st.session_state["active_result"] = res_model
                                    st.session_state["active_doc_id"] = doc["id"]
                                    st.session_state["active_document"] = None

                                    val_data = full_record.get("validation_data")
                                    if val_data:
                                        summary = ValidationSummary(
                                            total_metrics=val_data.get("total_metrics", 0),
                                            verified_count=val_data.get("verified_count", 0),
                                            flagged_count=val_data.get("flagged_count", 0),
                                            pending_count=val_data.get("pending_count", 0),
                                            contradictions=val_data.get("contradictions", []),
                                            duplicates=val_data.get("duplicates", []),
                                        )
                                        st.session_state["active_summary"] = summary

                                    st.success(f"Loaded '{doc['filename']}' into active session!")
                                    st.info("👉 Switch to the **Financial Extraction Results** tab to view.")
                                except Exception as load_exc:
                                    st.error(f"Failed deserializing extraction record: {load_exc}")

                    with btn_col2:
                        if st.button(
                            "🗑️ Delete Record",
                            key=f"del_doc_{doc['id']}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            repo.delete_document(doc["id"])
                            st.warning(f"Deleted document #{doc['id']} ('{doc['filename']}').")
                            st.rerun()

    # --------------------------------------------------------------------------
    # Tab 2: Cross-Document Trend Explorer
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.subheader("Cross-Document Metric Time-Series")
        st.caption("Compare extracted financial metrics across different reporting periods and documents.")

        metric_names = repo.get_distinct_metric_names()
        if not metric_names:
            st.info("No metrics extracted yet to chart trends.")
        else:
            default_metric = "Revenue" if "Revenue" in metric_names else metric_names[0]
            selected_trend_metric = st.selectbox(
                "Select Metric to Analyze:",
                options=metric_names,
                index=metric_names.index(default_metric),
            )

            hist_rows = repo.get_historical_metrics_by_name(selected_trend_metric)
            if hist_rows:
                df_hist = pd.DataFrame(hist_rows)
                st.dataframe(
                    df_hist[
                        [
                            "company_name",
                            "year",
                            "value",
                            "unit",
                            "reporting_period",
                            "verification_status",
                            "page_number",
                            "filename",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                # Render Plotly bar chart
                fig_trend = px.bar(
                    df_hist,
                    x="year",
                    y="value",
                    color="company_name",
                    barmode="group",
                    text="value",
                    title=f"Historical {selected_trend_metric} by Fiscal Year",
                    labels={"value": f"Value ({df_hist['unit'].iloc[0]})", "year": "Fiscal Year"},
                )
                fig_trend.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info(f"No numeric values available for metric '{selected_trend_metric}'.")

    # --------------------------------------------------------------------------
    # Tab 3: Pipeline Audit Logs
    # --------------------------------------------------------------------------
    with tab_logs:
        st.subheader("System & Pipeline Audit Trail")
        st.caption("Immutable record of document ingestions, LLM executions, and deletions.")

        logs = repo.get_audit_logs(limit=100)
        if logs:
            log_rows = []
            for l in logs:
                lvl = l["level"]
                icon = "ℹ️" if lvl == "INFO" else ("⚠️" if lvl == "WARNING" else "🚨")
                log_rows.append(
                    {
                        "Timestamp": l["created_at"][:19].replace("T", " "),
                        "Level": f"{icon} {lvl}",
                        "Event": l["event_type"],
                        "Document": l["filename"] or "-",
                        "Message": l["message"],
                    }
                )
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No audit log events recorded yet.")

    # --------------------------------------------------------------------------
    # Tab 4: Database Storage & Maintenance
    # --------------------------------------------------------------------------
    with tab_db:
        st.subheader("Database Health & Storage Metrics")
        stats = repo.db.get_database_stats()

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        with col_st1:
            st.metric("Database File Size", f"{stats['size_kb']} KB")
        with col_st2:
            st.metric("Stored Documents", stats["document_count"])
        with col_st3:
            st.metric("Stored Metrics", stats["metric_count"])
        with col_st4:
            st.metric("Audit Log Entries", stats["audit_log_count"])

        st.caption(f"File location: `{stats['database_path']}`")

        st.divider()

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🧹 Run Database Defragmentation (VACUUM)", use_container_width=True):
                repo.db.vacuum()
                st.success("Database vacuum completed successfully!")
                st.rerun()

        with col_act2:
            if st.button("⚠️ Clear All Historical Records", type="secondary", use_container_width=True):
                repo.clear_all_records()
                st.warning("All document and audit records cleared.")
                st.rerun()
