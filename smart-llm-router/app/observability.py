from datetime import datetime, timedelta, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models import RequestLog

TIMESERIES_MINUTES = 30


class RequestLogger:
    def log(
        self,
        db: Session,
        *,
        query_text: str,
        cache_hit: bool,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: float,
    ) -> None:
        db.add(
            RequestLog(
                query_text=query_text,
                cache_hit=cache_hit,
                model_used=model_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
            )
        )
        db.flush()

    def stats(self, db: Session) -> dict:
        totals = db.execute(
            select(
                func.count(RequestLog.id).label("total_requests"),
                func.coalesce(
                    func.avg(case((RequestLog.cache_hit.is_(True), 1.0), else_=0.0)),
                    0.0,
                ).label("cache_hit_rate"),
                func.coalesce(func.sum(RequestLog.estimated_cost_usd), 0.0).label(
                    "total_cost"
                ),
                func.coalesce(func.avg(RequestLog.latency_ms), 0.0).label(
                    "average_latency"
                ),
            )
        ).one()

        by_model_rows = db.execute(
            select(
                RequestLog.model_used,
                func.coalesce(func.sum(RequestLog.estimated_cost_usd), 0.0),
                func.count(RequestLog.id),
            ).group_by(RequestLog.model_used)
        ).all()

        cost_by_model = {
            model: {"cost_usd": float(cost), "requests": int(count)}
            for model, cost, count in by_model_rows
        }

        return {
            "total_requests": int(totals.total_requests),
            "cache_hit_rate": round(float(totals.cache_hit_rate), 4),
            "total_cost": round(float(totals.total_cost), 8),
            "average_latency": round(float(totals.average_latency), 2),
            "cost_by_model": cost_by_model,
            "requests_over_time": self._requests_over_time(db),
        }

    def _requests_over_time(self, db: Session) -> list[dict]:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        cutoff = now - timedelta(minutes=TIMESERIES_MINUTES - 1)
        bucket = func.date_trunc("minute", RequestLog.timestamp)
        rows = db.execute(
            select(
                bucket.label("bucket"),
                func.coalesce(
                    func.sum(case((RequestLog.cache_hit.is_(True), 1), else_=0)),
                    0,
                ).label("cache_hits"),
                func.coalesce(
                    func.sum(case((RequestLog.cache_hit.is_(False), 1), else_=0)),
                    0,
                ).label("llm_calls"),
            )
            .where(RequestLog.timestamp >= cutoff)
            .group_by(bucket)
            .order_by(bucket)
        ).all()

        by_minute: dict[datetime, tuple[int, int]] = {}
        for row in rows:
            key = row.bucket
            if key.tzinfo is None:
                key = key.replace(tzinfo=timezone.utc)
            by_minute[key.astimezone(timezone.utc).replace(second=0, microsecond=0)] = (
                int(row.cache_hits),
                int(row.llm_calls),
            )

        series = []
        for offset in range(TIMESERIES_MINUTES):
            minute = cutoff + timedelta(minutes=offset)
            cache_hits, llm_calls = by_minute.get(minute, (0, 0))
            series.append(
                {
                    "bucket": minute.isoformat(),
                    "cache_hits": cache_hits,
                    "llm_calls": llm_calls,
                }
            )
        return series
