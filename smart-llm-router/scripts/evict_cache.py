"""Hard-delete expired cache_entries rows.

lookup() already refuses to serve rows older than CACHE_TTL_DAYS, so this
script is just housekeeping — run it on a schedule (cron, a GitHub Actions
scheduled workflow, etc.) to keep the table and its HNSW index from growing
forever.

Usage:
    python scripts/evict_cache.py             # use CACHE_TTL_DAYS from .env
    python scripts/evict_cache.py --ttl-days 7  # override
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache import SemanticCache  # noqa: E402
from app.db import SessionLocal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=None,
        help="Override CACHE_TTL_DAYS from settings for this run.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        deleted = SemanticCache().evict_expired(db, ttl_days=args.ttl_days)
        db.commit()
        print(f"Evicted {deleted} expired cache entr{'y' if deleted == 1 else 'ies'}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
