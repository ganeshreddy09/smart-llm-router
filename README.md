# smart-llm-router

![tests](https://github.com/ganeshreddy09/smart-llm-router/actions/workflows/tests.yml/badge.svg)

FastAPI service that sits in front of an LLM API and demonstrates three MLOps patterns:

1. **Semantic caching** — skip the LLM when a close-enough prior query exists.
2. **Cost-aware routing** — send simple prompts to a cheap model and hard prompts to a larger one.
3. **Request observability** — persist per-request cost/latency and expose aggregates.

```
Client
  │  POST /query { "prompt": "..." }
  ▼
FastAPI
  │  MiniLM embedding (all-MiniLM-L6-v2, local)
  ▼
Postgres + pgvector cosine search
  │
  ├─ similarity > 0.92  → cache hit, cost $0, log + return
  └─ miss
       │  route_query(prompt)  → gpt-4o-mini | gpt-4o
       ▼
     OpenAI (or compatible) completion
       │  store embedding + response
       ▼
     request_logs row  → GET /stats
```
<img width="1198" height="953" alt="Screenshot 2026-10-02 185101" src="https://github.com/user-attachments/assets/50ebffcc-628c-411e-a88e-c682bc3435fa" />

Measured with python scripts/generate_traffic.py --repeat 3, run three times against a local docker compose up stack (23 unique prompts, 69 requests per run):

Metric	Result
Total requests	207
Cache hit rate	90.3%
Average latency	162 ms
Total cost	$0.0016
Requests by model	gpt-4o-mini 135, gpt-4o 72
Cost by model	gpt-4o $0.0014575, gpt-4o-mini $0.00010695

How to read these numbers:

Mock mode. These results were produced with MOCK_MODE=true, so no real LLM calls were made. Embeddings, pgvector search, routing, cost math and logging are real; the model response and its latency are simulated (see Mock mode). Costs are list prices applied to simulated token counts.
Synthetic traffic. The script repeats a small prompt set, so the hit rate is higher than real traffic would give. The first pass over a cold cache is mostly misses (71% hit rate, 283 ms average latency); the later runs hit the warm cache.
Routing impact. gpt-4o served about a third of requests but accounts for about 93% of spend, which is why routing simple prompts to gpt-4o-mini matters.

## Caching

Each prompt is encoded with [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` (384-d, runs locally, no embedding API bill). Vectors live in `cache_entries.embedding` (`vector(384)`).

Lookup uses pgvector cosine distance (`<=>`). Similarity is `1 - distance`. A **cache hit** is `similarity > CACHE_SIMILARITY_THRESHOLD` (default **0.92**). Hits return the stored answer, skip the LLM, and log `cache_hit=true` with `estimated_cost_usd=0`.

Misses call the LLM, then insert a new cache row so later paraphrases can hit.

## Routing

`route_query(query) -> model_name` in `app/routing.py` is the only decision point. Swap that function for a trained classifier without touching the API.

The current heuristic marks a prompt **complex** (default `gpt-4o`) if any of these hold:

- Keywords: `compare`, `explain why`, `step by step`, plus a few close variants
- Whitespace token count above `SIMPLE_MAX_TOKENS` (default 80)
- Character length above `SIMPLE_MAX_CHARS` (default 280)

Otherwise it is **simple** (default `gpt-4o-mini`).

Cost uses list prices in `app/config.py` (USD per token) times billed `usage` from the provider.

## API

| Method | Path | Body / result |
| --- | --- | --- |
| `POST` | `/query` | `{ "prompt": string }` → `{ "response", "cache_hit", "model_used", "latency_ms", "cost_usd" }` |
| `GET` | `/health` | `{ "status": "ok" }` |
| `GET` | `/stats` | `{ "total_requests", "cache_hit_rate", "total_cost", "average_latency", "cost_by_model", "requests_over_time" }` |
| `GET` | `/dashboard` | Static monitoring page (polls `/stats` every 5s) |

Every `/query` writes `request_logs`: timestamp, query text, cache hit, model, input/output tokens, estimated USD cost, latency in ms.

`/stats.requests_over_time` is the last 30 minutes, one bucket per minute (`cache_hits` vs `llm_calls`) so the dashboard can draw a stacked request chart without a second endpoint.

## Dashboard

Open http://localhost:8000/dashboard after `docker compose up`. The page is plain HTML/CSS/vanilla JS plus Chart.js (CDN), served from the FastAPI process via `StaticFiles`. It shows total cost, cache hit rate, average latency, requests over time (cache hits vs LLM calls), and cost by model.

## Mock mode

`MOCK_MODE=true` (the `.env.example` default) simulates LLM completions locally — no `OPENAI_API_KEY` needed, no API spend. Everything else in the pipeline still runs for real: MiniLM embeddings, pgvector similarity search, the routing heuristic, cost math against real list prices, and request logging. Only the actual model call (`app/llm.py`'s `MockLLMClient`) is faked — it returns canned text with token counts proportional to the prompt, and a short artificial delay (longer for the complex-model tier) so latency numbers on the dashboard still look realistic.

This is enough to exercise and screenshot the whole system — caching, routing, `/stats`, `/dashboard` — without spending anything. Set `MOCK_MODE=false` and a real `OPENAI_API_KEY` when you want numbers from an actual model for a resume claim.

## Auth

`POST /query` is **open by default** (no `API_KEY` set) — fine for running this locally as a portfolio demo. Set `API_KEY` in `.env` to require an `X-API-Key: <value>` header on every `/query` call; `/health`, `/stats`, and `/dashboard` stay unauthenticated since they don't spend money. This is a single `Depends()` check (see `require_api_key` in `app/main.py`), not a full auth system — deliberately minimal, and enough to stop this from being an open LLM proxy if it's ever deployed somewhere reachable from the internet.

## Cache expiry

`cache_entries` rows are TTL'd, not kept forever:

- `CACHE_TTL_DAYS` (default 30) controls the window.
- `SemanticCache.lookup()` filters `created_at` in the SQL query itself, so an expired row is never served even if it hasn't been deleted yet.
- `scripts/evict_cache.py` hard-deletes expired rows — run it on a schedule (cron, a scheduled GitHub Actions workflow, etc.) to keep the table and its HNSW index from growing unbounded:
  ```bash
  python scripts/evict_cache.py             # uses CACHE_TTL_DAYS from .env
  python scripts/evict_cache.py --ttl-days 7  # one-off override
  ```

## Run locally

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env (skip if MOCK_MODE=true, the .env.example default)

docker compose up --build
```

First build downloads the CPU-only PyTorch wheel (~200MB) plus MiniLM — a few minutes on a normal connection, longer on a slow one. This is a one-time cost; Docker caches the layer, so later builds skip it unless `requirements.txt` changes.

- API: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard
- Docs: http://localhost:8000/docs
- Postgres (pgvector): `localhost:5432` user/password/db `router` / `router` / `smart_llm_router`

The first image build downloads MiniLM so container start does not wait on Hugging Face.

Without Docker (Postgres with pgvector already running):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Generate real traffic (for resume numbers / dashboard screenshots)

Once `docker compose up` is running:

```bash
python scripts/generate_traffic.py
```

This sends a mix of simple factual prompts, complex multi-step prompts, and paraphrases of earlier prompts (to trigger real cache hits) at `/query`, then prints a `/stats` summary. Open `http://localhost:8000/dashboard` afterward for the numbers rendered as charts — that's your screenshot. Use `--repeat 3` for a bigger sample, or `--api-key` if you've set `API_KEY` in `.env`.

## Tests

```bash
pip install -r requirements.txt
pytest
```

The suite mocks LLM and database I/O — no Postgres needed to run it. It checks cache hit/miss, TTL enforcement and eviction, API-key auth, that routing picks the cheap vs large model, and that `/stats` returns the expected JSON shape. Runs automatically on every push/PR via `.github/workflows/tests.yml`.

## Project layout

```
app/
  config.py          env-driven settings (keys never hardcoded)
  db.py / models.py  SQLAlchemy + pgvector
  embeddings.py      MiniLM encoder
  cache.py           cosine lookup / store
  routing.py         swappable route_query()
  llm.py             OpenAI client + cost estimate
  observability.py   request_logs + /stats aggregates
  main.py            FastAPI routes
static/index.html    monitoring dashboard
sql/init.sql         extension + tables + HNSW index
scripts/
  evict_cache.py      hard-deletes expired cache_entries rows
  generate_traffic.py fires varied prompts at /query to seed real /stats data
.github/workflows/
  tests.yml           runs pytest on every push/PR
```
