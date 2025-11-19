import fitz
import pymupdf4llm

from rebelist.revelations.domain.exceptions import DocumentConverterError
from rebelist.revelations.domain.services import PdfConverterPort


class PdfConverter(PdfConverterPort):
    """Pdf to markdown converter."""

    def pdf_to_markdown(self, data: bytes) -> str:
        """Converts the raw binary content of a PDF document into a standardized Markdown formatted string."""
        try:
            with fitz.open('pdf', data) as document:
                return pymupdf4llm.to_markdown(
                    document, ignore_graphics=True, ignore_images=False, page_separators=False
                ).strip()
        except Exception as error:
            raise DocumentConverterError(f'Failed to convert PDF to Markdown: {error}') from error
