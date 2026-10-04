from unittest.mock import MagicMock, patch

from app.cache import CacheHit, SemanticCache, is_cache_hit
from app.config import get_settings


def test_is_cache_hit_above_threshold():
    assert is_cache_hit(0.921, threshold=0.92) is True
    assert is_cache_hit(0.92, threshold=0.92) is False
    assert is_cache_hit(0.5, threshold=0.92) is False


def _db_returning(row: dict | None) -> MagicMock:
    db = MagicMock()
    db.execute.return_value.mappings.return_value.first.return_value = row
    return db


@patch("app.cache.get_embedding", return_value=[0.1] * 384)
def test_lookup_returns_hit_when_similarity_exceeds_threshold(_embed):
    db = _db_returning(
        {
            "id": 1,
            "query_text": "what is caching?",
            "response": "storing answers for similar queries",
            "model_used": "gpt-4o-mini",
            "similarity": 0.97,
        }
    )
    hit = SemanticCache().lookup(db, "explain caching")
    assert hit == CacheHit(
        query_text="what is caching?",
        response="storing answers for similar queries",
        model_used="gpt-4o-mini",
        similarity=0.97,
    )


@patch("app.cache.get_embedding", return_value=[0.1] * 384)
def test_lookup_miss_when_similarity_is_too_low(_embed):
    db = _db_returning(
        {
            "id": 1,
            "query_text": "unrelated",
            "response": "nope",
            "model_used": "gpt-4o",
            "similarity": 0.4,
        }
    )
    assert SemanticCache().lookup(db, "something else") is None


@patch("app.cache.get_embedding", return_value=[0.1] * 384)
def test_lookup_miss_when_cache_empty(_embed):
    db = _db_returning(None)
    assert SemanticCache().lookup(db, "hello") is None


@patch("app.cache.get_embedding", return_value=[0.1] * 384)
def test_lookup_passes_configured_ttl_to_query(_embed):
    """The read-time expiry filter must use CACHE_TTL_DAYS, not a hardcoded value."""
    db = _db_returning(None)
    SemanticCache().lookup(db, "hello")
    _, params = db.execute.call_args.args
    assert params["ttl_days"] == get_settings().cache_ttl_days


def test_evict_expired_deletes_rows_older_than_ttl():
    db = MagicMock()
    db.execute.return_value.rowcount = 3

    deleted = SemanticCache().evict_expired(db, ttl_days=30)

    assert deleted == 3
    _, params = db.execute.call_args.args
    assert params["ttl_days"] == 30
    db.flush.assert_called_once()


def test_evict_expired_defaults_to_configured_ttl():
    db = MagicMock()
    db.execute.return_value.rowcount = 0

    SemanticCache().evict_expired(db)

    _, params = db.execute.call_args.args
    assert params["ttl_days"] == get_settings().cache_ttl_days
