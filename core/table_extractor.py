"""Tabular data extraction module for PDF (pdfplumber), DOCX, XLSX, and CSV."""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

logger = logging.getLogger("financial_intel.table_extractor")


@dataclass
class ExtractedTable:
    """Structured representation of an extracted tabular structure."""

    table_id: str
    page_number: int  # 1-indexed page or sheet index
    sheet_name: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    markdown: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert extracted table to a pandas DataFrame."""
        if not self.rows and not self.headers:
            return pd.DataFrame()
        if self.headers:
            return pd.DataFrame(self.rows, columns=self.headers)
        return pd.DataFrame(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted table to a serializable dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "sheet_name": self.sheet_name,
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "markdown": self.markdown,
            "bbox": self.bbox,
        }


def format_table_as_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Format headers and rows into clean GitHub-flavored markdown table string."""
    if not headers and not rows:
        return ""

    num_cols = len(headers) if headers else max((len(r) for r in rows), default=0)
    if num_cols == 0:
        return ""

    # Ensure headers list has appropriate length
    clean_headers = [str(h or "").strip().replace("\n", " ") for h in headers]
    if len(clean_headers) < num_cols:
        clean_headers.extend([f"Col_{i+1}" for i in range(len(clean_headers), num_cols)])

    clean_rows: List[List[str]] = []
    for row in rows:
        r = [str(c or "").strip().replace("\n", " ") for c in row]
        if len(r) < num_cols:
            r.extend([""] * (num_cols - len(r)))
        clean_rows.append(r[:num_cols])

    lines: List[str] = []
    lines.append("| " + " | ".join(clean_headers) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for r in clean_rows:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)


class TableExtractor:
    """Extracts tabular data across PDF, DOCX, XLSX, and CSV file formats."""

    def extract_pdf_tables(
        self,
        file_path: str | Path,
        page_numbers: Optional[List[int]] = None,
    ) -> List[ExtractedTable]:
        """Extract all detectable tables from a PDF using pdfplumber."""
        path = Path(file_path).resolve()
        tables: List[ExtractedTable] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    if page_numbers and page_num not in page_numbers:
                        continue

                    try:
                        extracted = page.extract_tables()
                        if not extracted:
                            continue

                        for t_idx, raw_table in enumerate(extracted):
                            if not raw_table:
                                continue

                            # Filter out completely empty rows
                            valid_rows = [
                                [str(cell).strip() if cell is not None else "" for cell in r]
                                for r in raw_table
                                if any(cell is not None and str(cell).strip() for cell in r)
                            ]
                            if not valid_rows:
                                continue

                            headers = valid_rows[0]
                            data_rows = valid_rows[1:] if len(valid_rows) > 1 else []
                            markdown = format_table_as_markdown(headers, data_rows)

                            table = ExtractedTable(
                                table_id=f"p{page_num}_t{t_idx+1}",
                                page_number=page_num,
                                headers=headers,
                                rows=data_rows,
                                num_rows=len(data_rows),
                                num_cols=len(headers),
                                markdown=markdown,
                            )
                            tables.append(table)
                    except Exception as page_exc:
                        logger.warning(
                            "Failed extracting tables from page %d of '%s': %s",
                            page_num,
                            path.name,
                            page_exc,
                        )
        except Exception as exc:
            logger.error("Failed opening PDF '%s' for table extraction: %s", path.name, exc)

        logger.info("Extracted %d tables from PDF '%s'", len(tables), path.name)
        return tables

    def extract_docx_tables(self, file_path: str | Path) -> List[ExtractedTable]:
        """Extract tables from a Microsoft Word (.docx) document."""
        import docx

        path = Path(file_path).resolve()
        tables: List[ExtractedTable] = []

        try:
            doc = docx.Document(str(path))
            for t_idx, docx_table in enumerate(doc.tables):
                raw_rows: List[List[str]] = []
                for row in docx_table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    if any(row_cells):
                        raw_rows.append(row_cells)

                if not raw_rows:
                    continue

                headers = raw_rows[0]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []
                markdown = format_table_as_markdown(headers, data_rows)

                tables.append(
                    ExtractedTable(
                        table_id=f"docx_t{t_idx+1}",
                        page_number=1,  # DOCX doesn't have native fixed page breaks without rendering
                        headers=headers,
                        rows=data_rows,
                        num_rows=len(data_rows),
                        num_cols=len(headers),
                        markdown=markdown,
                    )
                )
        except Exception as exc:
            logger.error("Failed extracting tables from DOCX '%s': %s", path.name, exc)

        logger.info("Extracted %d tables from DOCX '%s'", len(tables), path.name)
        return tables

    def extract_xlsx_tables(self, file_path: str | Path) -> List[ExtractedTable]:
        """Extract all worksheets and tabular data from an Excel (.xlsx) file."""
        path = Path(file_path).resolve()
        tables: List[ExtractedTable] = []

        try:
            excel_file = pd.ExcelFile(str(path))
            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Drop rows and columns that are completely null
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue

                headers = [str(col).strip() for col in df.columns]
                rows = [
                    [str(val).strip() if pd.notna(val) else "" for val in row]
                    for row in df.values.tolist()
                ]
                markdown = format_table_as_markdown(headers, rows)

                tables.append(
                    ExtractedTable(
                        table_id=f"sheet_{sheet_idx+1}",
                        page_number=sheet_idx + 1,
                        sheet_name=str(sheet_name),
                        headers=headers,
                        rows=rows,
                        num_rows=len(rows),
                        num_cols=len(headers),
                        markdown=markdown,
                    )
                )
        except Exception as exc:
            logger.error("Failed extracting sheets from XLSX '%s': %s", path.name, exc)

        logger.info("Extracted %d sheets/tables from XLSX '%s'", len(tables), path.name)
        return tables

    def extract_csv_table(self, file_path: str | Path) -> List[ExtractedTable]:
        """Read a CSV document and convert to a structured ExtractedTable."""
        path = Path(file_path).resolve()
        try:
            # Handle possible delimiter differences (comma, semicolon, tab)
            df = pd.read_csv(str(path), sep=None, engine="python")
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                return []

            headers = [str(col).strip() for col in df.columns]
            rows = [
                [str(val).strip() if pd.notna(val) else "" for val in row]
                for row in df.values.tolist()
            ]
            markdown = format_table_as_markdown(headers, rows)

            return [
                ExtractedTable(
                    table_id="csv_table_1",
                    page_number=1,
                    sheet_name=path.name,
                    headers=headers,
                    rows=rows,
                    num_rows=len(rows),
                    num_cols=len(headers),
                    markdown=markdown,
                )
            ]
        except Exception as exc:
            logger.error("Failed reading CSV '%s': %s", path.name, exc)
            return []
