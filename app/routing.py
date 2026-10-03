"""Swappable cost-aware routing. Replace `route_query` with a trained classifier later."""

from app.config import get_settings

COMPLEX_KEYWORDS = (
    "compare",
    "explain why",
    "step by step",
    "analyze",
    "analyse",
    "difference between",
    "pros and cons",
    "walk me through",
)


def estimate_tokens(text: str) -> int:
    """Whitespace token count — good enough as a complexity signal, not billed usage."""
    return max(1, len(text.split()))


def classify_complexity(query: str) -> str:
    lowered = query.lower()
    if any(keyword in lowered for keyword in COMPLEX_KEYWORDS):
        return "complex"

    settings = get_settings()
    if estimate_tokens(query) > settings.simple_max_tokens:
        return "complex"
    if len(query) > settings.simple_max_chars:
        return "complex"
    return "simple"


def route_query(query: str) -> str:
    """Return the model name for this query. Swap this function for an ML classifier."""
    settings = get_settings()
    if classify_complexity(query) == "complex":
        return settings.complex_model
    return settings.simple_model
