from rebelist.revelations.infrastructure.confluence.parsers import XHTMLParser


class TestXHTMLParser:
    """Test suite for the XHTMLParser class."""

    def test_parses_paragraphs(self) -> None:
        """Should convert paragraph tags to plain text with newline."""
        parser = XHTMLParser('<p>Hello <strong>world</strong>!</p>')
        assert parser.text() == 'Hello world!'

    def test_parses_headings(self) -> None:
        """Should convert heading tags to markdown-style headings."""
        parser = XHTMLParser('<h2>Section Title</h2>')
        assert parser.text().strip() == '## Section Title'

    def test_parses_lists(self) -> None:
        """Should convert ordered and unordered lists to markdown bullet points."""
        xhtml = """
        <ul>
            <li>First</li>
            <li>Second</li>
        </ul>
        <ol>
            <li>One</li>
            <li>Two</li>
        </ol>
        """
        parser = XHTMLParser(xhtml)
        result = parser.text()
        assert '- First' in result
        assert '- Second' in result
        assert '1. One' in result
        assert '1. Two' in result

    def test_parses_tables(self) -> None:
        """Should convert tables into pipe-delimited format."""
        xhtml = """
        <table>
            <tr><th>Header</th><th>Column</th></tr>
            <tr><td>Value1</td><td>Value2</td></tr>
        </table>
        """
        parser = XHTMLParser(xhtml)
        result = parser.text()
        assert '--- TABLE ---' in result
        assert 'Header | Column' in result
        assert 'Value1 | Value2' in result
        assert '--- END TABLE ---' in result

    def test_handles_structured_macro_info(self) -> None:
        """Should format info macro content."""
        xhtml = """
        <ac:structured-macro ac:name="info">
            <ac:rich-text-body><p>This is important.</p></ac:rich-text-body>
        </ac:structured-macro>
        """
        parser = XHTMLParser(xhtml)
        assert '[Note] This is important.' in parser.text()

    def test_handles_structured_macro_code(self) -> None:
        """Should format code blocks from macros."""
        xhtml = """
        <ac:structured-macro ac:name="code">
            <ac:plain-text-body><![CDATA[print("Hello")]]></ac:plain-text-body>
        </ac:structured-macro>
        """
        parser = XHTMLParser(xhtml)
        result = parser.text()
        assert '--- CODE ---' in result
        assert 'print("Hello")' in result
        assert '--- END CODE ---' in result

    def test_ignores_unknown_tags_but_preserves_text(self) -> None:
        """Unknown tags should not break parsing and still return inner content."""
        parser = XHTMLParser('<unknown><p>Text</p></unknown>')
        assert parser.text() == 'Text'

    def test_handles_nested_elements_gracefully(self) -> None:
        """Should flatten nested elements appropriately."""
        xhtml = """
        <div>
            <h3>Header</h3>
            <p><strong>Bold text</strong> and <em>emphasis</em>.</p>
        </div>
        """
        parser = XHTMLParser(xhtml)
        result = parser.text()
        assert '### Header' in result
        assert 'Bold text and emphasis.' in result
