"""Fire a realistic mix of prompts at the running service to generate real
/stats numbers — cache hit rate, cost by model, latency — instead of
guessing them for a resume bullet or README screenshot.

The prompt list is deliberately built from three groups:
  1. SIMPLE   — short factual questions. Should route to the cheap model.
  2. COMPLEX  — comparisons / multi-step asks. Should route to the big model.
  3. PARAPHRASES — reworded versions of earlier prompts, sent a second time.
     These are what should produce cache hits, since they're semantically
     close to something already asked.

Usage:
    python scripts/generate_traffic.py
    python scripts/generate_traffic.py --base-url http://localhost:8000 --repeat 2
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request

SIMPLE = [
    "What is the capital of France?",
    "What year did World War II end?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the largest planet in our solar system?",
    "What language is spoken in Brazil?",
    "What is the freezing point of water in Fahrenheit?",
    "Who painted the Mona Lisa?",
]

COMPLEX = [
    "Compare REST and GraphQL APIs for a mobile app backend.",
    "Explain why transformers need positional encodings.",
    "Walk me through how a binary search tree rebalances itself.",
    "What are the pros and cons of microservices versus a monolith?",
    "Analyze the tradeoffs between SQL and NoSQL databases for a chat app.",
    "Explain why garbage collection pauses happen and how generational GC helps.",
    "Compare Postgres and MongoDB for a project with relational data.",
    "Walk me through the steps of TLS handshake negotiation.",
]

# Reworded versions of a few SIMPLE prompts above — close enough in meaning
# that the semantic cache should catch them.
PARAPHRASES = [
    "Which city is the capital of France?",
    "In what year did the Second World War come to an end?",
    "At what temperature (Celsius) does water boil?",
    "Who is the author of Romeo and Juliet?",
    "What's gold's chemical symbol?",
]


def post_query(base_url: str, prompt: str, api_key: str | None) -> dict:
    body = json.dumps({"prompt": prompt}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(f"{base_url}/query", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_stats(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/stats") as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None, help="Only needed if API_KEY is set in .env")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Send each prompt this many times (default 1)"
    )
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between requests")
    args = parser.parse_args()

    prompts = (SIMPLE + COMPLEX + PARAPHRASES) * args.repeat
    random.shuffle(prompts)

    print(f"Sending {len(prompts)} prompts to {args.base_url}/query ...")
    ok, failed = 0, 0
    for i, prompt in enumerate(prompts, 1):
        try:
            result = post_query(args.base_url, prompt, args.api_key)
            tag = "HIT " if result["cache_hit"] else "MISS"
            print(f"[{i}/{len(prompts)}] {tag} model={result['model_used']:<12} "
                  f"cost=${result['cost_usd']:.6f}  {prompt[:60]}")
            ok += 1
        except urllib.error.URLError as exc:
            print(f"[{i}/{len(prompts)}] FAILED: {exc}")
            failed += 1
        time.sleep(args.delay)

    print(f"\nDone. {ok} succeeded, {failed} failed.")

    stats = get_stats(args.base_url)
    print("\n--- /stats summary ---")
    print(f"Total requests:     {stats['total_requests']}")
    print(f"Cache hit rate:     {stats['cache_hit_rate']:.1%}")
    print(f"Total cost:         ${stats['total_cost']:.6f}")
    print(f"Average latency:    {stats['average_latency']:.1f} ms")
    print("Cost by model:")
    for model, data in stats["cost_by_model"].items():
        print(f"  {model:<15} {data['requests']:>4} requests   ${data['cost_usd']:.6f}")
    print(f"\nOpen {args.base_url}/dashboard to see this rendered — that's your screenshot.")


if __name__ == "__main__":
    main()
