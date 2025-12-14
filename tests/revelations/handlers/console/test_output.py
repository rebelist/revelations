import pytest

from rebelist.revelations.handlers.console import Number


class TestNumber:
    """Test suite for the Number utility class."""

    PARAMS: list[tuple[float | int, Number.Scale, str]] = [
        # Percent scale
        (85, Number.Scale.PERCENT, '[green]85'),
        (70, Number.Scale.PERCENT, '[yellow]70'),
        (69.9, Number.Scale.PERCENT, '[red]69.9'),
        # Zero–one scale
        (0.9, Number.Scale.ZERO_ONE, '[green]0.9'),
        (0.7, Number.Scale.ZERO_ONE, '[yellow]0.7'),
        (0.699, Number.Scale.ZERO_ONE, '[red]0.7'),
        # One–five scale
        (5, Number.Scale.ONE_FIVE, '[green]5'),
        (4, Number.Scale.ONE_FIVE, '[yellow]4'),
        (3, Number.Scale.ONE_FIVE, '[red]3'),
    ]

    @pytest.mark.parametrize('value, scale, expected', PARAMS)
    def test_prettify_returns_expected_color_and_value(
        self,
        value: float | int,
        scale: Number.Scale,
        expected: str,
    ) -> None:
        """Tests that prettify returns the correct color tag and rounded value."""
        result = Number.prettify(value, scale)

        assert result == expected

    def test_prettify_rounds_value_to_two_decimals(self) -> None:
        """Tests that the prettified value is rounded to two decimal places."""
        result = Number.prettify(0.746, Number.Scale.ZERO_ONE)

        assert result == '[yellow]0.75'

    def test_prettify_handles_integer_values(self) -> None:
        """Tests that integer values are handled correctly."""
        result = Number.prettify(100, Number.Scale.PERCENT)

        assert result == '[green]100'
