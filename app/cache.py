from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings import get_embedding
from app.models import CacheEntry


@dataclass(frozen=True)
class CacheHit:
    query_text: str
    response: str
    model_used: str
    similarity: float


def is_cache_hit(similarity: float, threshold: float | None = None) -> bool:
    cutoff = get_settings().cache_similarity_threshold if threshold is None else threshold
    return similarity > cutoff


class SemanticCache:
    def lookup(self, db: Session, prompt: str) -> CacheHit | None:
        embedding = get_embedding(prompt)
        settings = get_settings()
        threshold = settings.cache_similarity_threshold
        # pgvector cosine distance (<=>) is 1 - cosine similarity for normalized vectors.
        # created_at filter enforces the TTL at read time, independent of
        # whether the eviction job has run yet — a stale row is never served
        # even if it hasn't been physically deleted.
        # psycopg2 sends a plain Python list as a numeric[] array literal, and
        # Postgres has no implicit numeric[] -> vector conversion — pgvector's
        # <=> operator only matches vector <=> vector, so the cast to ::vector
        # is required here, not optional. Without it this raises
        # "operator does not exist: vector <=> numeric[]" on every request.
        stmt = text(
            """
            SELECT id, query_text, response, model_used,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM cache_entries
            WHERE created_at > NOW() - (:ttl_days || ' days')::interval
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 1
            """
        )
        row = db.execute(
            stmt,
            {
                "embedding": "[" + ",".join(str(float(x)) for x in embedding) + "]",
                "ttl_days": settings.cache_ttl_days,
            },
        ).mappings().first()
        if row is None:
            return None
        similarity = float(row["similarity"])
        if not is_cache_hit(similarity, threshold):
            return None
        return CacheHit(
            query_text=row["query_text"],
            response=row["response"],
            model_used=row["model_used"],
            similarity=similarity,
        )

    def store(self, db: Session, prompt: str, response: str, model_used: str) -> None:
        entry = CacheEntry(
            query_text=prompt,
            embedding=get_embedding(prompt),
            response=response,
            model_used=model_used,
        )
        db.add(entry)
        db.flush()

    def evict_expired(self, db: Session, ttl_days: int | None = None) -> int:
        """Hard-delete cache rows older than the TTL. Returns rows deleted.

        lookup() already refuses to *serve* expired rows, so this is purely
        housekeeping — it keeps cache_entries (and the HNSW index built on
        it) from growing forever. Safe to run on a schedule (see
        scripts/evict_cache.py) or by hand.
        """
        days = get_settings().cache_ttl_days if ttl_days is None else ttl_days
        stmt = text(
            "DELETE FROM cache_entries WHERE created_at <= NOW() - (:ttl_days || ' days')::interval"
        )
        result = db.execute(stmt, {"ttl_days": days})
        db.flush()
        return result.rowcount