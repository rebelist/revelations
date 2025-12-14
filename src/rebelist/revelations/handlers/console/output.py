from enum import Enum


class Number:
    class Scale(Enum):
        """Represents a numeric scale."""

        PERCENT = 'percent'
        ZERO_ONE = 'zero_one'
        ONE_FIVE = 'one_five'

    @staticmethod
    def prettify(value: float | int, scale: 'Number.Scale') -> str:
        """Format a value with a color based on its normalized score."""
        score = Number._normalize(value, scale)
        rounded_value = round(value, 2)
        color = Number._color_from_score(score)
        return f'[{color}]{rounded_value}'

    @staticmethod
    def _normalize(value: float | int, scale: 'Number.Scale') -> float:
        """Normalize a value to a 0–1 score based on its scale."""
        match scale:
            case Number.Scale.PERCENT:
                return value / 100.0
            case Number.Scale.ZERO_ONE:
                return float(value)
            case Number.Scale.ONE_FIVE:
                return (value - 1) / 4

    @staticmethod
    def _color_from_score(score: float) -> str:
        """Pick a color for a normalized score using threshold rules."""
        if score >= 0.85:
            return 'green'
        if score >= 0.7:
            return 'yellow'
        return 'red'
