from datetime import datetime
from typing import List

import pytest
import pytest_mock

from rebelist.revelations.domain import ContextDocument
from rebelist.revelations.infrastructure.ollama.adapters import OllamaAdapter


@pytest.fixture
def sample_documents() -> List[ContextDocument]:
    """Provides a list of example context documents for testing."""
    return [
        ContextDocument(
            title='Doc A',
            content='Alpha content',
            modified_at=datetime(2024, 2, 15, 10, 30, 0),
        ),
        ContextDocument(
            title='Doc B',
            content='Beta content',
            modified_at=datetime(2024, 2, 15, 10, 30, 0),
        ),
    ]


def test_respond_returns_expected_response(
    mocker: pytest_mock.MockerFixture,
    sample_documents: list[ContextDocument],
) -> None:
    """Ensures OllamaAdapter respond builds the chain and invokes it with the correct context."""
    mock_chain = mocker.MagicMock()
    mock_chain.invoke.return_value = 'Mocked Answer'

    mock_prompt_template = mocker.MagicMock()
    mock_prompt_template_cls = mocker.patch('rebelist.revelations.infrastructure.ollama.adapters.PromptTemplate')
    mock_prompt_template_cls.from_template.return_value = mock_prompt_template

    mock_output_parser = mocker.MagicMock()
    mocker.patch(
        'rebelist.revelations.infrastructure.ollama.adapters.StrOutputParser',
        return_value=mock_output_parser,
    )

    mock_llm = mocker.MagicMock()
    mock_prompt_template.__or__.return_value = mocker.MagicMock()
    mock_prompt_template.__or__.return_value.__or__.return_value = mock_chain

    mocker.patch(
        'rebelist.revelations.infrastructure.ollama.adapters.ResponseGeneratorPort.get_prompt',
        return_value='{question}\n{context}',
    )

    adapter = OllamaAdapter(mock_llm)
    response = adapter.respond('What is RAG?', sample_documents)

    expected_context = '## Document: Doc A\n\nAlpha content\n\n## Document: Doc B\n\nBeta content'

    mock_chain.invoke.assert_called_once_with(
        {
            'question': 'What is RAG?',
            'context': expected_context,
        }
    )

    assert response.answer == 'Mocked Answer'
    assert list(response.documents) == sample_documents
