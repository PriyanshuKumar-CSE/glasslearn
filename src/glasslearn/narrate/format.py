"""Turning numbers into short, readable text."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

_SCALES = ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"))


def number(x: float, digits: int = 3) -> str:
    """Format a number for a sentence, with commas above 1,000.

    >>> number(1234.5), number(12.345), number(0.012345), number(2.5e9)
    ('1,234', '12.3', '0.0123', '2.5 billion')
    """
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    if abs(x) >= 1e15:
        return f"{x:.{digits - 1}e}"
    for size, word in _SCALES:
        if abs(x) >= size:
            return f"{number(x / size, digits)} {word}"
    if abs(x) >= 1000:
        decimals = max(0, digits - (int(math.log10(abs(x))) + 1))
        return f"{x:,.{decimals}f}"
    if abs(x) < 1e-4:
        return f"{x:.{digits - 1}e}"
    text = f"{x:.{digits}g}"
    if "e" in text:  # rounding reached 1,000 (999.7 -> 1e+03)
        text = f"{x:,.0f}"
    return text


def distinct(
    a: float, b: float, digits: int = 3, max_digits: int = 6
) -> tuple[str, str]:
    """Format two numbers with just enough digits to tell them apart.

    Numbers that agree to ``max_digits`` significant digits are shown with
    ``digits`` and come out equal, which callers treat as "unchanged".

    >>> distinct(542.31, 541.87)
    ('542.3', '541.9')
    """
    for d in range(digits, max_digits + 1):
        left, right = number(a, d), number(b, d)
        if left != right:
            return left, right
    return number(a, digits), number(b, digits)


def signed(x: float, digits: int = 3) -> str:
    """Like :func:`number`, with a leading + for positive values."""
    text = number(x, digits)
    return text if text.startswith("-") or x == 0 else f"+{text}"


def percent(fraction: float) -> str:
    """Format a fraction as a whole percentage: 0.853 -> '85%'."""
    return f"{round(100 * fraction):d}%"


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 point', '3 points'; pass ``many`` for irregular plurals."""
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def join_words(words: Iterable[str], last: str = "and") -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(words)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {last} {items[-1]}"


def numbers(values: Sequence[float]) -> str:
    """'12, 15 and 20': a short list of formatted numbers."""
    return join_words(number(v) for v in values)


def capitalize(text: str) -> str:
    """Upper-case the first letter only (keeps names like 'k-means' intact)."""
    return text[:1].upper() + text[1:]
