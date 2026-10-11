"""One-line definitions added the first time a term appears in a narration."""

from __future__ import annotations

import re

#: term -> definition. Terms are matched as whole words, ignoring case.
TERMS: dict[str, str] = {
    "centroid": "the centre of a cluster, at the average position of its points.",
    "inertia": (
        "the sum of squared distances from every point to its centroid; "
        "lower means tighter clusters."
    ),
    "k-means++": (
        "a way to choose starting centroids that are spread out, which usually "
        "means fewer iterations."
    ),
    "iteration": "one assignment pass followed by one update pass.",
    "tolerance": (
        "how small the centroid moves must get before training counts as finished."
    ),
    "restart": (
        "a separate run from different starting centroids; the best one is kept."
    ),
    "empty cluster": "a cluster that ended up with no points assigned to it.",
}

_PATTERNS = {
    term: re.compile(rf"(?<![\w-]){re.escape(term)}s?(?![\w-])", re.IGNORECASE)
    for term in TERMS
}


def terms_in(text: str) -> list[str]:
    """Return the glossary terms used in ``text``, in glossary order."""
    return [term for term, pattern in _PATTERNS.items() if pattern.search(text)]


def define(term: str) -> str:
    """Return 'Term: definition' for a glossary term."""
    return f"{term[:1].upper()}{term[1:]}: {TERMS[term]}"
