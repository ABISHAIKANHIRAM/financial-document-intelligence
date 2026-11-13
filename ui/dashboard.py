"""Dashboard UI view: high-level KPIs, visual analytics, and recent activity."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.repository import DocumentRepository


def render_dashboard(repo: DocumentRepository) -> None:
    """Render the executive financial intelligence dashboard."""
    st.header("Executive Intelligence Dashboard")
    st.caption("Overview of local document processing volume, extraction KPIs, and validation quality.")

    kpis = repo.get_kpis()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Processed Documents", kpis["total_documents"])
    with c2:
        st.metric("Total Extracted Metrics", kpis["total_metrics"])
    with c3:
        st.metric("Verified Metrics", kpis["verified_metrics"])
    with c4:
        st.metric("Metrics Requiring Review", kpis["flagged_metrics"])

    st.divider()

    # Visualizations using Plotly
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Verification Quality Distribution")
        total = kpis["total_metrics"]
        if total > 0:
            labels = ["Verified", "Flagged (Review Required)", "Pending"]
            values = [
                kpis["verified_metrics"],
                kpis["flagged_metrics"],
                kpis["pending_metrics"],
            ]
            colors = ["#22c55e", "#ef4444", "#3b82f6"]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.45,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No financial metrics extracted yet. Upload a document to view analytics.")

    with col_chart2:
        st.subheader("Document Volume by Format")
        recent_docs = repo.list_documents(limit=50)
        if recent_docs:
            df_docs = pd.DataFrame(recent_docs)
            type_counts = df_docs["file_type"].value_counts().reset_index()
            type_counts.columns = ["Format", "Count"]

            fig_bar = px.bar(
                type_counts,
                x="Format",
                y="Count",
                color="Format",
                text="Count",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No documents ingested yet.")

    st.subheader("Recent Document Ingestion History")
    if recent_docs:
        table_rows = []
        for d in recent_docs[:10]:
            table_rows.append(
                {
                    "ID": d["id"],
                    "Document Name": d["filename"],
                    "Format": d["file_type"].upper(),
                    "Company": d["company_name"] or "-",
                    "Fiscal Year": d["financial_year"] or "-",
                    "Total Metrics": d["metric_count"],
                    "Verified": d["verified_count"] or 0,
                    "Flagged": d["flagged_count"] or 0,
                    "Processed At": d["created_at"][:19].replace("T", " "),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No historical records found.")
