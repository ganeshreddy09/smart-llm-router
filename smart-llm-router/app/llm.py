from dataclasses import dataclass

import random
import time

from openai import OpenAI

from app.config import get_settings


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    def complete(self, prompt: str, model: str) -> LLMResult:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResult(
            text=choice,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class MockLLMClient:
    """Drop-in replacement for LLMClient — no network call, no API key, no cost.

    Enabled by MOCK_MODE=true. Everything upstream and downstream of this
    class still runs for real: MiniLM embeddings, pgvector similarity
    lookup, the routing heuristic, cost math against real list prices, and
    request logging. Only the actual model call is faked, so /stats numbers
    are synthetic (they show what the pipeline *would* cost/measure against
    a real model, using its real token counts and pricing) rather than
    numbers from a live model — good enough to exercise and screenshot the
    system end to end without spending anything.
    """

    _TEMPLATES = (
        "Here's a concise answer: {gist}.",
        "Based on general knowledge, {gist}.",
        "In short, {gist} — let me know if you want more detail.",
        "{gist}. That covers the key point.",
    )

    def complete(self, prompt: str, model: str) -> LLMResult:
        settings = get_settings()
        # Simulate the larger model being slower than the cheap one, so
        # latency numbers on the dashboard look realistic rather than ~0ms.
        if model == settings.complex_model:
            time.sleep(random.uniform(0.5, 1.1))
        else:
            time.sleep(random.uniform(0.1, 0.3))

        gist = prompt.strip().rstrip("?.!")[:80]
        text = random.choice(self._TEMPLATES).format(gist=gist)

        # Synthetic but proportional to real input, so cost/latency scale
        # with prompt length the way a real model's usage would.
        input_tokens = max(1, len(prompt.split()))
        output_tokens = max(1, len(text.split()))
        return LLMResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    settings = get_settings()
    pricing = {
        settings.simple_model: (
            settings.gpt4o_mini_input_per_token,
            settings.gpt4o_mini_output_per_token,
        ),
        settings.complex_model: (
            settings.gpt4o_input_per_token,
            settings.gpt4o_output_per_token,
        ),
    }
    input_rate, output_rate = pricing.get(
        model,
        (settings.gpt4o_mini_input_per_token, settings.gpt4o_mini_output_per_token),
    )
    return round(input_tokens * input_rate + output_tokens * output_rate, 8)
