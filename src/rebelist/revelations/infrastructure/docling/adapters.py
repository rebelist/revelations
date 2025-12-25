import io

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

from rebelist.revelations.domain.exceptions import DocumentConverterError
from rebelist.revelations.domain.services import PdfConverterPort


class PdfConverter(PdfConverterPort):
    """PDF to Markdown converter."""

    def __init__(self, converter: DocumentConverter):
        self.__converter = converter

    def pdf_to_markdown(self, data: bytes) -> str:
        """Converts the raw binary content of a PDF document into a standardized Markdown formatted string."""
        try:
            stream = io.BytesIO(data)
            doc_stream = DocumentStream(name='Tmp', stream=stream)
            result = self.__converter.convert(doc_stream)

            markdown = result.document.export_to_markdown()

            return markdown.strip()

        except Exception as error:
            raise DocumentConverterError(f'Failed to convert PDF to Markdown: {error}') from error
