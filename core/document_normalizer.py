"""Content normalization module for financial document text and tabular integration."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from core.table_extractor import ExtractedTable
from core.text_extractor import PageText


class DocumentNormalizer:
    """Cleans, normalizes, and integrates raw document text and extracted tables."""

    LIGATURE_MAP = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }

    UNICODE_REPLACEMENTS = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "—": " - ",
        "–": " - ",
        "−": "-",  # unicode minus sign
        "…": "...",
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow no-break space
        "\u2009": " ",  # thin space
        "\u200b": "",   # zero-width space
        "\t": "    ",
    }

    def normalize_text(self, text: str) -> str:
        """Apply deterministic cleaning to raw text strings."""
        if not text:
            return ""

        # Normalize unicode to NFKC
        result = unicodedata.normalize("NFKC", text)

        # Replace ligatures
        for lig, repl in self.LIGATURE_MAP.items():
            result = result.replace(lig, repl)

        # Replace quotation marks, special dashes, and non-standard spaces
        for target, repl in self.UNICODE_REPLACEMENTS.items():
            result = result.replace(target, repl)

        # Standardize multiple blank lines to double newline
        result = re.sub(r"\r\n|\r", "\n", result)
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()

    def integrate_page_content(
        self,
        page_text: PageText,
        page_tables: Optional[List[ExtractedTable]] = None,
    ) -> str:
        """Combine normalized narrative text and extracted markdown tables for a page.

        Provides explicit markdown delimiters around tables so LLMs can distinguish tabular columns.
        """
        normalized_narrative = self.normalize_text(page_text.text)
        sections: List[str] = []

        if normalized_narrative:
            sections.append(normalized_narrative)

        if page_tables:
            for table in page_tables:
                if table.markdown.strip():
                    table_title = (
                        f"[TABLE {table.table_id}: {table.sheet_name or f'Page {table.page_number}'}]"
                    )
                    table_block = (
                        f"{table_title}\n"
                        f"{table.markdown.strip()}\n"
                        f"[/TABLE {table.table_id}]"
                    )
                    sections.append(table_block)

        return "\n\n".join(sections).strip()

    @staticmethod
    def normalize_financial_number(raw_val: str) -> Optional[float]:
        """Parse raw financial string (accounting parentheses, thousand separators) to float.

        Examples:
            "(1,250.5)" -> -1250.5
            "€ 45.2M"   -> 45200000.0 (multiplier handled if suffix present)
            "-15.4%"    -> -15.4
        """
        if not raw_val or not raw_val.strip():
            return None

        clean = raw_val.strip().replace("$", "").replace("€", "").replace("£", "").replace("EUR", "").replace("USD", "")
        clean = clean.replace("\u00a0", " ").strip()

        is_negative = False
        # Check accounting brackets: (1,234.56)
        if clean.startswith("(") and clean.endswith(")"):
            is_negative = True
            clean = clean[1:-1].strip()
        elif clean.startswith("-"):
            is_negative = True
            clean = clean[1:].strip()

        # Check multiplier suffix
        multiplier = 1.0
        upper_clean = clean.upper()
        if upper_clean.endswith("B") or upper_clean.endswith("BN"):
            multiplier = 1_000_000_000.0
            clean = re.sub(r"(?i)bn|b", "", clean).strip()
        elif upper_clean.endswith("M") or upper_clean.endswith("MN"):
            multiplier = 1_000_000.0
            clean = re.sub(r"(?i)mn|m", "", clean).strip()
        elif upper_clean.endswith("K"):
            multiplier = 1_000.0
            clean = re.sub(r"(?i)k", "", clean).strip()
        elif upper_clean.endswith("%"):
            clean = clean[:-1].strip()

        # Handle European decimal notation (1.234,56) vs US/UK (1,234.56)
        if "," in clean and "." in clean:
            if clean.rfind(",") > clean.rfind("."):
                # European: 1.234,56
                clean = clean.replace(".", "").replace(",", ".")
            else:
                # US/UK: 1,234.56
                clean = clean.replace(",", "")
        elif "," in clean:
            # Ambiguous: could be 1,000 or 1,5
            # If 3 digits follow comma, assume thousand separator
            parts = clean.split(",")
            if len(parts) == 2 and len(parts[1]) == 3:
                clean = clean.replace(",", "")
            elif len(parts) > 2:
                clean = clean.replace(",", "")
            else:
                clean = clean.replace(",", ".")

        try:
            val = float(clean) * multiplier
            return -val if is_negative else val
        except ValueError:
            return None
