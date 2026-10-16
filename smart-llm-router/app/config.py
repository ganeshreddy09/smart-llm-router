from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://router:router@localhost:5432/smart_llm_router"
    openai_api_key: str = ""
    openai_base_url: str | None = None

    cache_similarity_threshold: float = 0.92
    embedding_model: str = "all-MiniLM-L6-v2"

    # How long a cache entry stays servable before it's treated as stale.
    # Enforced two ways: lookup() filters expired rows at read time, and
    # scripts/evict_cache.py hard-deletes them to keep the table small.
    cache_ttl_days: int = 30

    simple_model: str = "gpt-4o-mini"
    complex_model: str = "gpt-4o"

    simple_max_tokens: int = 80
    simple_max_chars: int = 280

    # USD per token (list prices; override via env if they change)
    gpt4o_mini_input_per_token: float = 0.15 / 1_000_000
    gpt4o_mini_output_per_token: float = 0.60 / 1_000_000
    gpt4o_input_per_token: float = 2.50 / 1_000_000
    gpt4o_output_per_token: float = 10.00 / 1_000_000

    log_level: str = "INFO"

    # If set, POST /query requires header "X-API-Key: <this value>".
    # Left empty by default so local/demo use needs no setup — see README
    # "Auth" section before exposing this anywhere but localhost.
    api_key: str = ""

    # When true, LLM completions are simulated locally (canned text,
    # synthetic token counts, artificial latency) — no OPENAI_API_KEY and
    # no API spend needed. Embeddings/caching/routing/cost-math all still
    # run for real. See README "Mock mode".
    mock_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
