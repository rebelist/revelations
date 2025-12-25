from typing import Any

import pytest
from docling.document_converter import DocumentConverter as DoclingConverter
from pytest_mock import MockerFixture

from rebelist.revelations.domain.exceptions import DocumentConverterError
from rebelist.revelations.infrastructure.docling.adapters import PdfConverter


class TestPdfConverter:
    """Tests for the PdfConverter class."""

    @pytest.fixture
    def mock_pdf_data(self) -> bytes:
        """Mock PDF binary content (random bytes)."""
        # Create 100 random bytes to simulate PDF data
        return b'\x89PDF\x01' + b'A' * 95 + b'EOF'

    def test_pdf_to_markdown_success(self, mocker: MockerFixture, mock_pdf_data: bytes):
        """Test successful conversion of PDF bytes to Markdown string."""
        docling = mocker.create_autospec(DoclingConverter, spec_set=True, instance=True)
        result = mocker.Mock()
        result.document.export_to_markdown.return_value = '## Sample Markdown'
        docling.convert.return_value = result

        converter = PdfConverter(docling)
        result = converter.pdf_to_markdown(mock_pdf_data)

        assert result == '## Sample Markdown'
        docling.convert.assert_called_once()

    def test_pdf_to_markdown_failure(self, mocker: Any, mock_pdf_data: bytes):
        """Test that failure in conversion raises DocumentConverterError."""
        docling = mocker.create_autospec(DoclingConverter, spec_set=True, instance=True)
        docling.convert.side_effect = DocumentConverterError('PDF data is corrupted')
        converter = PdfConverter(docling)

        with pytest.raises(DocumentConverterError) as excinfo:
            converter.pdf_to_markdown(mock_pdf_data)

        assert 'Failed to convert PDF to Markdown' in str(excinfo.value)
        assert 'PDF data is corrupted' in str(excinfo.value)
