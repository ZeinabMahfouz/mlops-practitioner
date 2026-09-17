"""Step 7c: CAT 5 -- vLLM.

Serves a small instruct model (Qwen2.5-1.5B-Instruct) through vLLM's
OpenAI-compatible endpoint and measures TTFT (time to first token) and
inter-token latency at increasing concurrency, using the same OpenAI
client code you'd point at a hosted API.

GPU CONSTRAINT: this machine has no NVIDIA GPU (confirmed via `nvidia-smi`
inside WSL -- no passthrough device present), and vLLM's GPU backend is
the whole point of the exercise (PagedAttention / continuous batching
need a GPU to demonstrate meaningfully; vLLM's CPU backend exists but
doesn't exercise the same scheduler). Per the course's own prerequisite
note for GPU-dependent steps, this was run as a notebook exercise on
Google Colab's free T4 GPU tier instead -- documented here as the honest
engineering constraint it is, same treatment as Step 7a would get.

To reproduce on a machine with a GPU (or in Colab):
    pip install vllm openai
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 &
    # wait for http://localhost:8000/health to return 200, then:
    python scripts/benchmark_cat5_vllm.py
"""

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Explain why p95 latency matters more than average latency for a production API.",
    "What is the difference between batch inference and online inference?",
    "Describe what a dead-letter queue is used for in a streaming pipeline.",
    "Why does the GIL limit concurrency in a Python web server?",
    "What is the tradeoff between max batch size and max latency in model serving?",
    "Explain what a consumer group does in a message streaming system.",
    "Why would you choose a canary release over a blue/green deployment?",
    "What does p50 versus p99 latency tell you about a system?",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CAT 5 benchmark: vLLM TTFT / inter-token latency"
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--max-tokens", type=int, default=100)
    return p.parse_args(argv)


async def one_request(client: AsyncOpenAI, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    n_tokens = 0

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            n_tokens += 1

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else None
    inter_token_ms = (
        (last_token_time - first_token_time) * 1000 / max(n_tokens - 1, 1)
        if first_token_time and n_tokens > 1
        else None
    )
    return {"ttft_ms": ttft_ms, "n_tokens": n_tokens, "inter_token_ms": inter_token_ms}


async def run_concurrency_level(
    client: AsyncOpenAI, concurrency: int, n_requests: int, max_tokens: int
) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def bound_request(i):
        async with sem:
            return await one_request(client, PROMPTS[i % len(PROMPTS)], max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*[bound_request(i) for i in range(n_requests)])
    wall_time = time.perf_counter() - start

    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    inter = [r["inter_token_ms"] for r in results if r["inter_token_ms"] is not None]

    def pct(data, p):
        s = sorted(data)
        return s[int(len(s) * p)] if data else None

    print(
        f"\n=== Concurrency {concurrency} ({n_requests} requests, wall time {wall_time:.2f}s) ==="
    )
    print(
        f"TTFT       mean={statistics.mean(ttfts):.1f}ms  p50={pct(ttfts,0.5):.1f}ms  p95={pct(ttfts,0.95):.1f}ms"
    )
    print(
        f"Inter-tok  mean={statistics.mean(inter):.2f}ms  p50={pct(inter,0.5):.2f}ms  p95={pct(inter,0.95):.2f}ms"
    )
    print(f"Throughput: {n_requests / wall_time:.2f} requests/sec")
    return {
        "concurrency": concurrency,
        "wall_time": wall_time,
        "ttfts": ttfts,
        "inter": inter,
    }


async def main_async(args) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed")
    for concurrency in args.concurrency_levels:
        n_requests = max(concurrency * 3, 10)
        await run_concurrency_level(client, concurrency, n_requests, args.max_tokens)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
