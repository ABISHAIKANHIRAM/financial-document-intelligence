"""Application configuration and local model settings UI view."""

from __future__ import annotations

import streamlit as st

from config import Settings
from core.ocr_processor import OcrProcessor
from llm.ollama_client import OllamaClient


def render_settings(settings: Settings, client: OllamaClient, ocr: OcrProcessor) -> None:
    """Render system settings, local Ollama diagnostics, and processing configurations."""
    st.header("System Settings & Local AI Configuration")
    st.caption("Configure local Ollama connection, inference parameters, chunk sizes, and OCR engine.")

    # 1. Ollama Status & Diagnostics
    st.subheader("Local LLM (Ollama) Status")
    status = client.get_status()

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Ollama Service", "🟢 Online" if status.is_running else "🔴 Offline")
    with col_s2:
        st.metric("Configured Model", status.configured_model)
    with col_s3:
        st.metric("Model Available", "Yes" if status.model_available else "No")

    if not status.is_running:
        st.error(
            f"Could not connect to Ollama at `{status.base_url}`. "
            "Ensure Ollama is started locally by running `ollama serve` in your terminal."
        )
    elif not status.model_available:
        st.warning(
            f"Configured model `{status.configured_model}` is not downloaded locally. "
            f"Install it by running `ollama pull {status.configured_model}`."
        )

    st.divider()

    # 2. Model & Inference Settings
    st.subheader("Inference & Local Model Parameters")

    installed = client.list_models()
    model_choices = [m.name for m in installed] if installed else [settings.ollama_model]
    curr_idx = (
        model_choices.index(client.resolve_model_name(settings.ollama_model, installed))
        if client.resolve_model_name(settings.ollama_model, installed) in model_choices
        else 0
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        new_model = st.selectbox(
            "Default Local Model",
            options=model_choices,
            index=curr_idx,
            help="Select the model for financial intelligence extraction.",
        )
        new_url = st.text_input(
            "Ollama Base URL",
            value=settings.ollama_base_url,
            help="Local host endpoint strictly required (e.g. http://localhost:11434).",
        )
    with col_m2:
        new_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.ollama_temperature),
            step=0.05,
            help="Lower values (0.0) produce deterministic, fact-grounded extractions.",
        )
        new_ctx = st.select_slider(
            "Context Window (num_ctx)",
            options=[2048, 4096, 8192, 16384, 32768],
            value=settings.ollama_num_ctx,
            help="Ollama context token length for analyzing financial documents.",
        )

    st.divider()

    # 3. Chunking & Document Processing Settings
    st.subheader("Document Chunking & Processing")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        new_chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=1000,
            max_value=16000,
            value=settings.chunk_size,
            step=500,
            help="Maximum characters per document chunk.",
        )
    with col_c2:
        new_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=2000,
            value=settings.chunk_overlap,
            step=50,
            help="Character overlap between consecutive chunks.",
        )
    with col_c3:
        new_max_chunks = st.number_input(
            "Max Chunks per Document",
            min_value=5,
            max_value=200,
            value=settings.max_chunks_per_document,
            step=5,
        )

    st.divider()

    # 4. OCR & Tesseract Settings
    st.subheader("OCR Engine & PDF Rasterization")

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        new_ocr_enabled = st.checkbox(
            "Enable OCR Fallback for Scanned Pages",
            value=settings.ocr_enabled,
        )
        ocr_ready = ocr.is_available()
        st.write(f"**Tesseract Status:** {'🟢 Ready' if ocr_ready else '⚠️ Not Found'}")
    with col_o2:
        new_ocr_lang = st.text_input(
            "OCR Language",
            value=settings.ocr_language,
            help="Language code passed to Tesseract (e.g. eng, deu, fra).",
        )
    with col_o3:
        new_dpi = st.select_slider(
            "OCR Rasterization DPI",
            options=[150, 200, 300, 400],
            value=settings.ocr_dpi,
            help="Resolution for rendering PDF pages before OCR.",
        )

    new_tess_cmd = st.text_input(
        "Tesseract Binary Path Override (optional)",
        value=settings.tesseract_cmd or "",
        help="Full path to tesseract.exe if not on PATH.",
    )

    if st.button("Save Settings", type="primary", use_container_width=True):
        # Update settings object in-memory
        settings.ollama_model = new_model
        settings.ollama_base_url = new_url
        settings.ollama_temperature = new_temp
        settings.ollama_num_ctx = new_ctx
        settings.chunk_size = new_chunk_size
        settings.chunk_overlap = new_overlap
        settings.max_chunks_per_document = new_max_chunks
        settings.ocr_enabled = new_ocr_enabled
        settings.ocr_language = new_ocr_lang
        settings.ocr_dpi = new_dpi
        settings.tesseract_cmd = new_tess_cmd or None

        st.success("Settings updated successfully!")
        st.rerun()
