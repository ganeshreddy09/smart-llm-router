from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    prompt: str = Field(min_length=1)


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    model_used: str
    latency_ms: float
    cost_usd: float


class HealthResponse(BaseModel):
    status: str


class ModelCost(BaseModel):
    cost_usd: float
    requests: int


class RequestTimeBucket(BaseModel):
    bucket: str
    cache_hits: int
    llm_calls: int


class StatsResponse(BaseModel):
    total_requests: int
    cache_hit_rate: float
    total_cost: float
    average_latency: float
    cost_by_model: dict[str, ModelCost]
    requests_over_time: list[RequestTimeBucket]
