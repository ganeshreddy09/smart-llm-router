import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.cache import SemanticCache
from app.config import get_settings
from app.db import get_db, init_db
from app.llm import LLMClient, MockLLMClient, estimate_cost_usd
from app.observability import RequestLogger
from app.routing import estimate_tokens, route_query
from app.schemas import HealthResponse, QueryRequest, QueryResponse, StatsResponse

logger = logging.getLogger("smart-llm-router")
logging.basicConfig(level=get_settings().log_level)

cache = SemanticCache()
llm_client = MockLLMClient() if get_settings().mock_mode else LLMClient()
request_logger = RequestLogger()
if get_settings().mock_mode:
    logger.warning("MOCK_MODE is on — /query responses are simulated, no OpenAI calls are made.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="smart-llm-router",
    description="Semantic cache + cost-aware LLM routing with request observability.",
    lifespan=lifespan,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op when API_KEY is unset (local/demo default).

    Set API_KEY in .env to require "X-API-Key: <value>" on POST /query —
    intended for the moment this stops being localhost-only. See README.
    """
    configured_key = get_settings().api_key
    if not configured_key:
        return
    if x_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)) -> StatsResponse:
    return StatsResponse.model_validate(request_logger.stats(db))


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(body: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    started = time.perf_counter()
    prompt = body.prompt

    hit = cache.lookup(db, prompt)
    if hit is not None:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info("cache_hit similarity=%.4f model=%s", hit.similarity, hit.model_used)
        request_logger.log(
            db,
            query_text=prompt,
            cache_hit=True,
            model_used=hit.model_used,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(hit.response),
            estimated_cost_usd=0.0,
            latency_ms=latency_ms,
        )
        return QueryResponse(
            response=hit.response,
            cache_hit=True,
            model_used=hit.model_used,
            latency_ms=round(latency_ms, 2),
            cost_usd=0.0,
        )

    model = route_query(prompt)
    logger.info("cache_miss routing_to=%s", model)
    result = llm_client.complete(prompt, model)
    cost = estimate_cost_usd(model, result.input_tokens, result.output_tokens)
    cache.store(db, prompt, result.text, model)

    latency_ms = (time.perf_counter() - started) * 1000
    request_logger.log(
        db,
        query_text=prompt,
        cache_hit=False,
        model_used=model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost_usd=cost,
        latency_ms=latency_ms,
    )
    return QueryResponse(
        response=result.text,
        cache_hit=False,
        model_used=model,
        latency_ms=round(latency_ms, 2),
        cost_usd=cost,
    )


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/dashboard", include_in_schema=True)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
