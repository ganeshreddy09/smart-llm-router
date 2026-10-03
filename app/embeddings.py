from functools import lru_cache

from app.config import get_settings


@lru_cache
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def get_embedding(text: str) -> list[float]:
    """Encode query text with MiniLM (384-d). Isolated so tests can mock it."""
    vector = _load_model().encode(text, normalize_embeddings=True)
    return vector.tolist()
