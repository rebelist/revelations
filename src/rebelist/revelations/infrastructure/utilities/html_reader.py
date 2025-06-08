from bs4 import BeautifulSoup, Tag


class HTMLParser:
    """Parse HTML documents."""

    @staticmethod
    def text(html: str) -> str:
        """Extracts text from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup(['script', 'style']):
                tag.decompose()

            structured_text: list[str] = []

            for tag in soup.find_all(['p', 'ul', 'ol', 'table']):
                if isinstance(tag, Tag):
                    if tag.name in ['ul', 'ol']:
                        list_items: list[str] = [
                            f'- {li.get_text(strip=True)}' for li in tag.find_all('li') if isinstance(li, Tag)
                        ]
                        structured_text.append('\n'.join(list_items))
                    elif tag.name == 'table':
                        rows: list[str] = []
                        for row in tag.find_all('tr'):
                            if isinstance(row, Tag):
                                columns: list[str] = [
                                    col.get_text(strip=True)
                                    for col in row.find_all(['th', 'td'])
                                    if isinstance(col, Tag)
                                ]
                                rows.append(' | '.join(columns))
                        structured_text.append('\n'.join(rows))
                    else:
                        structured_text.append(tag.get_text(strip=True))

            return '\n'.join(structured_text)

        except (AttributeError, TypeError, ValueError):
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(strip=True)
