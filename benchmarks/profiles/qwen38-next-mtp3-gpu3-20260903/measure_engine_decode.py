#!/usr/bin/env python3
"""Measure one decode workload through an OpenAI-compatible completions API.

The output separates user-visible token throughput from target-model verifier
throughput. It recognizes the native speculative-decode counters exposed by
vLLM and SGLang and records the counter selected for each server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

VERIFY_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "sglang:spec_verify_calls_total",
)


def get_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=10) as response:
        lines = response.read().decode().splitlines()
    values: dict[str, float] = {}
    for name in VERIFY_COUNTERS:
        values[name] = sum(
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.startswith(name + "{") or line.startswith(name + " ")
        )
    return values


def median(runs: list[dict[str, object]], key: str) -> float:
    return statistics.median(float(run[key]) for run in runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how speculative decoding improves language model inference "
            "throughput, including its correctness requirement."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    # A zero counter is valid before the first request. Select by probing the
    # metric text when numeric values alone cannot distinguish availability.
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
        metric_text = response.read().decode()
    counter = next((name for name in VERIFY_COUNTERS if name in metric_text), None)
    if counter is None:
        raise RuntimeError("server exposes no supported speculative verifier counter")

    runs: list[dict[str, object]] = []
    for run_index in range(args.warmups + args.samples):
        payload: dict[str, object] = {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "seed": args.seed,
            "ignore_eos": True,
        }
        before = metrics(f"{base_url}/metrics")[counter]
        started = time.perf_counter()
        response = get_json(f"{base_url}/v1/completions", payload)
        elapsed = time.perf_counter() - started
        after = metrics(f"{base_url}/metrics")[counter]
        verifier_rounds = after - before
        usage = response.get("usage", {})
        output_tokens = int(usage.get("completion_tokens", 0))
        text = "".join(str(choice.get("text", "")) for choice in response["choices"])
        item = {
            "run": run_index,
            "warmup": run_index < args.warmups,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": output_tokens,
            "wall_seconds": elapsed,
            "output_tokens_per_second": output_tokens / elapsed,
            "verifier_rounds": verifier_rounds,
            "verifier_rounds_per_second": verifier_rounds / elapsed,
            "accepted_tokens_per_round": output_tokens / verifier_rounds,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        runs.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)

    measured = [run for run in runs if not run["warmup"]]
    result = {
        "base_url": base_url,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "sampling": {"temperature": 0, "top_p": 1, "top_k": -1},
        "seed": args.seed,
        "verifier_counter": counter,
        "warmups": args.warmups,
        "samples": args.samples,
        "median_output_tokens_per_second": median(measured, "output_tokens_per_second"),
        "median_verifier_rounds_per_second": median(
            measured, "verifier_rounds_per_second"
        ),
        "median_accepted_tokens_per_round": median(
            measured, "accepted_tokens_per_round"
        ),
        "runs": runs,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
