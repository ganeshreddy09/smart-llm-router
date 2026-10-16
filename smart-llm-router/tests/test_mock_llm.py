from unittest.mock import patch

from app.config import get_settings
from app.llm import MockLLMClient


@patch("app.llm.time.sleep")  # don't actually wait during tests
def test_mock_complete_returns_nonempty_result(_sleep):
    result = MockLLMClient().complete("What is the capital of France?", "gpt-4o-mini")
    assert result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0


@patch("app.llm.time.sleep")
def test_mock_complete_scales_tokens_with_prompt_length(_sleep):
    short = MockLLMClient().complete("Hi", "gpt-4o-mini")
    long_prompt = "word " * 50
    long = MockLLMClient().complete(long_prompt, "gpt-4o-mini")
    assert long.input_tokens > short.input_tokens


@patch("app.llm.time.sleep")
def test_mock_complete_simulates_longer_delay_for_complex_model(mock_sleep):
    settings = get_settings()
    MockLLMClient().complete("hard question", settings.complex_model)
    simple_delay_call = mock_sleep.call_args.args[0]
    assert 0.5 <= simple_delay_call <= 1.1

    MockLLMClient().complete("easy question", settings.simple_model)
    cheap_delay_call = mock_sleep.call_args.args[0]
    assert 0.1 <= cheap_delay_call <= 0.3
