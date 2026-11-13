"""Financial Document Intelligence - Streamlit Web Application."""

from __future__ import annotations

import streamlit as st

from config import Settings, configure_logging, get_settings
from core.document_loader import DocumentLoader
from core.ocr_processor import OcrProcessor
from database.db import DatabaseManager
from database.repository import DocumentRepository
from llm.financial_extractor import FinancialExtractor
from llm.ollama_client import OllamaClient
from ui.dashboard import render_dashboard
from ui.history import render_history
from ui.results import render_results
from ui.settings import render_settings
from ui.upload import render_upload


def main() -> None:
    """Main application lifecycle and page routing."""
    st.set_page_config(
        page_title="Financial Document Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = get_settings()
    configure_logging(settings)

    # Initialize shared dependencies in session state
    if "db_manager" not in st.session_state:
        st.session_state["db_manager"] = DatabaseManager(settings=settings)
    if "repository" not in st.session_state:
        st.session_state["repository"] = DocumentRepository(st.session_state["db_manager"])
    if "ollama_client" not in st.session_state:
        st.session_state["ollama_client"] = OllamaClient()
    if "ocr_processor" not in st.session_state:
        st.session_state["ocr_processor"] = OcrProcessor(settings=settings)
    if "document_loader" not in st.session_state:
        st.session_state["document_loader"] = DocumentLoader(
            settings=settings, ocr_processor=st.session_state["ocr_processor"]
        )
    if "financial_extractor" not in st.session_state:
        st.session_state["financial_extractor"] = FinancialExtractor(
            ollama_client=st.session_state["ollama_client"], settings=settings
        )
    if "financial_validator" not in st.session_state:
        from validation.financial_validator import FinancialValidator

        st.session_state["financial_validator"] = FinancialValidator()

    repo: DocumentRepository = st.session_state["repository"]
    loader: DocumentLoader = st.session_state["document_loader"]
    extractor: FinancialExtractor = st.session_state["financial_extractor"]
    validator = st.session_state["financial_validator"]
    client: OllamaClient = st.session_state["ollama_client"]
    ocr: OcrProcessor = st.session_state["ocr_processor"]

    # Sidebar Navigation
    with st.sidebar:
        st.title("📈 Financial Intel")
        st.caption("Local AI-Powered Document Extraction")

        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "📊 Dashboard",
                "📤 Upload Documents",
                "📋 Financial Extraction Results",
                "🗄️ Document History",
                "⚙️ Settings",
            ],
            index=0,
        )

        st.divider()

        # Privacy guarantee badge
        st.success("🔒 **Local Execution**\n\nAll documents and LLM inferences remain strictly on this local machine.")

        # Quick Ollama connectivity indicator
        ollama_online = client.is_running()
        if ollama_online:
            st.caption(f"🟢 Ollama: `{client.base_url}`")
        else:
            st.caption(f"🔴 Ollama Offline (`{client.base_url}`)")

    # Route to selected view
    if page == "📊 Dashboard":
        render_dashboard(repo)
    elif page == "📤 Upload Documents":
        render_upload(repo, loader, extractor, validator, settings)
    elif page == "📋 Financial Extraction Results":
        render_results()
    elif page == "🗄️ Document History":
        render_history(repo)
    elif page == "⚙️ Settings":
        render_settings(settings, client, ocr)


if __name__ == "__main__":
    main()
