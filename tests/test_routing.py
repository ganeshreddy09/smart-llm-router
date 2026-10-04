from app.config import get_settings
from app.routing import classify_complexity, route_query


def test_simple_factual_query_routes_to_cheap_model():
    prompt = "What is the capital of France?"
    assert classify_complexity(prompt) == "simple"
    assert route_query(prompt) == get_settings().simple_model


def test_keyword_compare_routes_to_complex_model():
    prompt = "Compare Postgres and MySQL for analytics workloads."
    assert classify_complexity(prompt) == "complex"
    assert route_query(prompt) == get_settings().complex_model


def test_explain_why_keyword_is_complex():
    prompt = "Explain why transformers need positional encodings."
    assert route_query(prompt) == get_settings().complex_model


def test_step_by_step_keyword_is_complex():
    prompt = "Solve this step by step: 12 * 8 + 3"
    assert route_query(prompt) == get_settings().complex_model


def test_long_query_routes_to_complex_model():
    prompt = " ".join(["detail"] * 90)
    assert classify_complexity(prompt) == "complex"
    assert route_query(prompt) == get_settings().complex_model
