from unittest.mock import MagicMock, create_autospec

import pytest
from docling.document_converter import DocumentConverter as DoclingConverter

from rebelist.revelations.domain.exceptions import DocumentConverterError
from rebelist.revelations.infrastructure.docling.adapters import PdfConverter


class TestPdfConverter:
    """Tests for the PdfConverter class."""

    @pytest.fixture
    def mock_pdf_data(self) -> bytes:
        """Mock PDF binary content (random bytes)."""
        return b'\x89PDF\x01' + b'A' * 95 + b'EOF'

    def test_pdf_to_markdown_success(self, mock_pdf_data: bytes) -> None:
        """Test successful conversion of PDF bytes to Markdown string."""
        docling = create_autospec(DoclingConverter, spec_set=True, instance=True)
        result = MagicMock()
        result.document.export_to_markdown.return_value = '## Sample Markdown'
        docling.convert.return_value = result

        converter = PdfConverter(docling)
        output = converter.pdf_to_markdown(mock_pdf_data)

        assert output == '## Sample Markdown'
        docling.convert.assert_called_once()

    def test_pdf_to_markdown_failure(self, mock_pdf_data: bytes) -> None:
        """Test that failure in conversion raises DocumentConverterError."""
        docling = create_autospec(DoclingConverter, spec_set=True, instance=True)
        docling.convert.side_effect = DocumentConverterError('PDF data is corrupted')
        converter = PdfConverter(docling)

        with pytest.raises(DocumentConverterError) as excinfo:
            converter.pdf_to_markdown(mock_pdf_data)

        assert 'Failed to convert PDF to Markdown' in str(excinfo.value)
        assert 'PDF data is corrupted' in str(excinfo.value)
