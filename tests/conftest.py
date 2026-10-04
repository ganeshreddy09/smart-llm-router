import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://router:router@localhost:5432/smart_llm_router"
)
os.environ.setdefault("CACHE_SIMILARITY_THRESHOLD", "0.92")
os.environ.setdefault("SIMPLE_MODEL", "gpt-4o-mini")
os.environ.setdefault("COMPLEX_MODEL", "gpt-4o")
