from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rebelist.revelations.domain.exceptions import DocumentConverterError
from rebelist.revelations.infrastructure.mupdf.adapters import PdfConverter


class TestPdfConverter:
    """Tests for the PdfConverter class."""

    @pytest.fixture
    def mock_pdf_data(self) -> bytes:
        """Mock PDF binary content (random bytes)."""
        # Create 100 random bytes to simulate PDF data
        return b'\x89PDF\x01' + b'A' * 95 + b'EOF'

    def test_pdf_to_markdown_success(self, mocker: MockerFixture, mock_pdf_data: bytes):
        """Test successful conversion of PDF bytes to markdown string."""
        mock_to_markdown = mocker.patch('pymupdf4llm.to_markdown', return_value='## Sample Markdown')
        mock_fitz_doc = MagicMock()
        mock_fitz_open = mocker.patch('fitz.open', return_value=mock_fitz_doc)

        converter = PdfConverter()
        result = converter.pdf_to_markdown(mock_pdf_data)

        assert result == '## Sample Markdown'
        mock_fitz_open.assert_called_once_with('pdf', mock_pdf_data)
        mock_to_markdown.assert_called_once_with(
            mock_fitz_doc.__enter__.return_value,
            ignore_graphics=True,
            ignore_images=False,
            page_separators=False,
        )

    def test_pdf_to_markdown_failure(self, mocker: Any, mock_pdf_data: bytes):
        """Test that failure in conversion raises DocumentConverterError."""
        mock_fitz_open = mocker.patch('fitz.open')
        mock_fitz_open.side_effect = RuntimeError('PDF data is corrupted')

        converter = PdfConverter()

        with pytest.raises(DocumentConverterError) as excinfo:
            converter.pdf_to_markdown(mock_pdf_data)

        assert 'Failed to convert PDF to Markdown' in str(excinfo.value)
        assert 'PDF data is corrupted' in str(excinfo.value)
