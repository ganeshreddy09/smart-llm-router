CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cache_entries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    response TEXT NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS cache_entries_embedding_hnsw_idx
    ON cache_entries
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_text TEXT NOT NULL,
    cache_hit BOOLEAN NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_cost_usd NUMERIC(12, 8) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS request_logs_timestamp_idx ON request_logs (timestamp);
CREATE INDEX IF NOT EXISTS request_logs_model_used_idx ON request_logs (model_used);
