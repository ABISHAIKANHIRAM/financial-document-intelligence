"""Document upload, inspection, and extraction execution UI view."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from config import Settings, get_settings
from core.document_loader import DocumentLoader, ProcessedDocument
from database.repository import DocumentRepository
from llm.financial_extractor import FinancialExtractor
from llm.ollama_client import OllamaClient
from validation.financial_validator import FinancialValidator

logger = logging.getLogger("financial_intel.ui_upload")


def render_upload(
    repo: DocumentRepository,
    loader: DocumentLoader,
    extractor: FinancialExtractor,
    validator: FinancialValidator,
    settings: Settings,
) -> None:
    """Render the document upload and interactive extraction view."""
    st.header("Document Ingestion & Financial Extraction")
    st.caption("Upload annual reports, quarterly releases, financial models, or press releases.")

    uploaded_file = st.file_uploader(
        "Choose a financial document (PDF, DOCX, XLSX, CSV)",
        type=["pdf", "docx", "xlsx", "xls", "csv"],
        help="Uploads are processed 100% locally on your machine. No data is sent to external cloud APIs.",
    )

    if uploaded_file is None:
        st.info("Supported formats: **PDF** (selectable or scanned), **DOCX**, **XLSX**, and **CSV**.")
        return

    # Display file metadata
    file_bytes = uploaded_file.getvalue()
    file_size_kb = round(len(file_bytes) / 1024, 2)
    suffix = Path(uploaded_file.name).suffix.lower()

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.text_input("Filename", uploaded_file.name, disabled=True)
    with col_info2:
        st.text_input("Format", suffix.upper().replace(".", ""), disabled=True)
    with col_info3:
        st.text_input("Size", f"{file_size_kb} KB", disabled=True)

    # Extraction options
    client = extractor.client
    installed_models = client.list_models()
    model_names = [m.name for m in installed_models] if installed_models else [client.model]
    default_idx = (
        model_names.index(client.resolve_model_name(client.model, installed_models))
        if client.resolve_model_name(client.model, installed_models) in model_names
        else 0
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        selected_model = st.selectbox(
            "Local Ollama Model",
            options=model_names,
            index=default_idx,
            help="Select the locally running model to perform financial information extraction.",
        )
    with col_opt2:
        skip_ocr = st.checkbox(
            "Disable OCR Fallback for this upload",
            value=not settings.ocr_enabled,
            help="Check to skip OCR on scanned PDF pages (faster, but may skip image-only text).",
        )

    start_btn = st.button("🚀 Process & Extract Financial Data", type="primary", use_container_width=True)

    if start_btn:
        # Save file to upload directory
        upload_path = settings.upload_dir / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(file_bytes)

        st.divider()
        status_container = st.status("Running Financial Document Pipeline...", expanded=True)

        try:
            # 1. Document Loading & Inspection
            status_container.write("Step 1/4: Ingesting and inspecting document structure...")
            doc = loader.load(upload_path)
            repo.log_pipeline_event(
                event_type="document_inspection",
                message=f"Ingested '{uploaded_file.name}' ({doc.file_type}): {doc.page_count} pages, {len(doc.tables)} tables.",
                level="INFO",
                details={"page_count": doc.page_count, "table_count": len(doc.tables)},
            )
            status_container.write(
                f"✓ Document ingested: {doc.page_count} pages/sheets, {len(doc.tables)} tables detected, {len(doc.chunks)} chunks generated."
            )
            if doc.inspection_report and doc.inspection_report.scanned_page_numbers:
                status_container.write(
                    f"⚠️ Scanned pages detected: {doc.inspection_report.scanned_page_numbers}"
                )

            # 2. Local LLM Financial Extraction
            status_container.write(
                f"Step 2/4: Performing local AI extraction using model `{selected_model}`..."
            )
            progress_bar = st.progress(0.0)

            def update_progress(current: int, total: int, msg: str) -> None:
                progress_bar.progress(min(1.0, current / max(1, total)))
                status_container.write(msg)

            raw_result = extractor.extract_from_document(
                document=doc,
                model=selected_model,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0)
            status_container.write(
                f"✓ LLM extraction finished: {len(raw_result.metrics)} financial metrics retrieved."
            )

            # 3. Deterministic Validation & Source Traceability Verification
            status_container.write(
                "Step 3/4: Applying deterministic accounting rules and verifying source evidence..."
            )
            validated_result, validation_summary = validator.validate_document_results(
                raw_result, document=doc
            )
            status_container.write(
                f"✓ Validation complete: {validation_summary.verified_count} verified, {validation_summary.flagged_count} require review."
            )

            # 4. Persistence to SQLite
            status_container.write("Step 4/4: Saving extraction record to local SQLite database...")
            doc_id = repo.save_extraction_pipeline_output(
                document=doc,
                extraction_result=validated_result,
                validation_summary=validation_summary,
            )
            repo.log_pipeline_event(
                event_type="extraction_complete",
                message=f"Pipeline finished for '{uploaded_file.name}': {len(validated_result.metrics)} metrics.",
                document_id=doc_id,
                level="INFO",
                details={
                    "verified": validation_summary.verified_count,
                    "flagged": validation_summary.flagged_count,
                    "model": selected_model,
                },
            )
            status_container.write(f"✓ Saved to database with Record ID #{doc_id}.")
            status_container.update(label="Extraction Pipeline Succeeded!", state="complete")

            # Store active result in session state
            st.session_state["active_doc_id"] = doc_id
            st.session_state["active_document"] = doc
            st.session_state["active_result"] = validated_result
            st.session_state["active_summary"] = validation_summary

            st.success(
                f"Successfully extracted {len(validated_result.metrics)} financial metrics from **{uploaded_file.name}**!"
            )
            st.info("👉 Switch to the **Financial Extraction Results** or **JSON Viewer** tab to inspect and export.")

        except Exception as exc:
            logger.exception("Document processing failed: %s", exc)
            status_container.update(label="Pipeline Failed", state="error")
            st.error(f"Error during document processing: {exc}")
            if doc := locals().get("doc"):
                if doc.errors:
                    st.write("**Document Errors:**", doc.errors)
