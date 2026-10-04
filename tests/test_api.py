from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.cache import CacheHit
from app.config import get_settings
from app.db import get_db
from app.llm import LLMResult
from app.main import app, cache, llm_client, request_logger


@pytest.fixture
def client():
    def _fake_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = _fake_db
    with patch("app.main.init_db"):
        with TestClient(app) as test_client:
            yield test_client
    app.dependency_overrides = {}


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_cache_hit_skips_llm(client: TestClient):
    hit = CacheHit(
        query_text="prior",
        response="cached answer",
        model_used="gpt-4o-mini",
        similarity=0.99,
    )
    with (
        patch.object(cache, "lookup", return_value=hit),
        patch.object(llm_client, "complete") as complete,
        patch.object(request_logger, "log") as log,
    ):
        response = client.post("/query", json={"prompt": "What is caching?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "cached answer"
    assert body["cache_hit"] is True
    assert body["model_used"] == "gpt-4o-mini"
    assert body["cost_usd"] == 0.0
    assert "latency_ms" in body
    complete.assert_not_called()
    log.assert_called_once()
    assert log.call_args.kwargs["cache_hit"] is True


def test_query_cache_miss_uses_routed_model(client: TestClient):
    result = LLMResult(text="Paris is the capital.", input_tokens=10, output_tokens=6)
    with (
        patch.object(cache, "lookup", return_value=None),
        patch.object(cache, "store") as store,
        patch.object(llm_client, "complete", return_value=result) as complete,
        patch.object(request_logger, "log") as log,
    ):
        response = client.post(
            "/query", json={"prompt": "What is the capital of France?"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["cache_hit"] is False
    assert body["model_used"] == "gpt-4o-mini"
    assert body["response"] == "Paris is the capital."
    assert body["cost_usd"] > 0
    complete.assert_called_once()
    assert complete.call_args.args[1] == "gpt-4o-mini"
    store.assert_called_once()
    assert log.call_args.kwargs["cache_hit"] is False


def test_query_cache_miss_routes_complex_prompt(client: TestClient):
    result = LLMResult(text="A long comparison...", input_tokens=40, output_tokens=80)
    with (
        patch.object(cache, "lookup", return_value=None),
        patch.object(cache, "store"),
        patch.object(llm_client, "complete", return_value=result) as complete,
        patch.object(request_logger, "log"),
    ):
        response = client.post(
            "/query",
            json={"prompt": "Compare REST and GraphQL and explain why one might fit."},
        )

    assert response.status_code == 200
    assert response.json()["model_used"] == "gpt-4o"
    assert complete.call_args.args[1] == "gpt-4o"


def test_stats_shape(client: TestClient):
    payload = {
        "total_requests": 4,
        "cache_hit_rate": 0.25,
        "total_cost": 0.0012,
        "average_latency": 123.45,
        "cost_by_model": {
            "gpt-4o-mini": {"cost_usd": 0.0002, "requests": 3},
            "gpt-4o": {"cost_usd": 0.001, "requests": 1},
        },
        "requests_over_time": [
            {
                "bucket": "2026-09-27T08:00:00+00:00",
                "cache_hits": 1,
                "llm_calls": 3,
            }
        ],
    }
    with patch.object(request_logger, "stats", return_value=payload):
        response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "total_requests",
        "cache_hit_rate",
        "total_cost",
        "average_latency",
        "cost_by_model",
        "requests_over_time",
    }
    assert isinstance(body["total_requests"], int)
    assert isinstance(body["cache_hit_rate"], float)
    assert isinstance(body["total_cost"], float)
    assert isinstance(body["average_latency"], float)
    assert "gpt-4o-mini" in body["cost_by_model"]
    assert set(body["cost_by_model"]["gpt-4o-mini"].keys()) == {"cost_usd", "requests"}
    assert body["requests_over_time"][0]["cache_hits"] == 1
    assert body["requests_over_time"][0]["llm_calls"] == 3


def test_query_works_without_api_key_when_unset(client: TestClient):
    """Default config: API_KEY="" means /query stays open, no header needed."""
    assert get_settings().api_key == ""
    hit = CacheHit(query_text="q", response="a", model_used="gpt-4o-mini", similarity=0.99)
    with patch.object(cache, "lookup", return_value=hit), patch.object(request_logger, "log"):
        response = client.post("/query", json={"prompt": "hi"})
    assert response.status_code == 200


def test_query_rejects_missing_key_when_configured(client: TestClient):
    fake_settings = MagicMock(api_key="secret-123")
    with patch("app.main.get_settings", return_value=fake_settings):
        response = client.post("/query", json={"prompt": "hi"})
    assert response.status_code == 401


def test_query_rejects_wrong_key_when_configured(client: TestClient):
    fake_settings = MagicMock(api_key="secret-123")
    with patch("app.main.get_settings", return_value=fake_settings):
        response = client.post(
            "/query", json={"prompt": "hi"}, headers={"X-API-Key": "wrong"}
        )
    assert response.status_code == 401


def test_query_accepts_correct_key_when_configured(client: TestClient):
    fake_settings = MagicMock(api_key="secret-123")
    hit = CacheHit(query_text="q", response="a", model_used="gpt-4o-mini", similarity=0.99)
    with (
        patch("app.main.get_settings", return_value=fake_settings),
        patch.object(cache, "lookup", return_value=hit),
        patch.object(request_logger, "log"),
    ):
        response = client.post(
            "/query", json={"prompt": "hi"}, headers={"X-API-Key": "secret-123"}
        )
    assert response.status_code == 200


def test_health_and_stats_do_not_require_api_key(client: TestClient):
    """Auth only guards the paid endpoint — health checks and dashboards stay open."""
    fake_settings = MagicMock(api_key="secret-123")
    with patch("app.main.get_settings", return_value=fake_settings):
        assert client.get("/health").status_code == 200
        with patch.object(request_logger, "stats", return_value={
            "total_requests": 0, "cache_hit_rate": 0.0, "total_cost": 0.0,
            "average_latency": 0.0, "cost_by_model": {}, "requests_over_time": [],
        }):
            assert client.get("/stats").status_code == 200


def test_dashboard_serves_html(client: TestClient):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "requests-chart" in html
    assert "cost-chart" in html
    assert "/stats" in html
    assert "5000" in html
