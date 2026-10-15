"""Prompt sources: generators yielding (prompt_text, max_tokens).

`fixed_prompts` pins both lengths for curve-mapping (CONCEPTS.md section 7).
`sample_prompts` cycles through a bundled 300-row file with a ShareGPT-like
length distribution. The file is synthetic (lognormal prompt and output lengths,
generated with a fixed seed), chosen over a real dataset to keep the package tiny
and licence-free; its job is to provide realistic *shape*, not realistic content.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from importlib import resources

_FILLER = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "while", "the", "engineer", "measures", "latency", "throughput", "memory", "and", "cost", "under", "steadily", "increasing", "load"]


def make_prompt(approx_tokens: int) -> str:
    """Roughly `approx_tokens` tokens of English (about one token per word)."""
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(approx_tokens))


def fixed_prompts(prompt_tokens: int, max_tokens: int) -> Iterator[tuple[str, int]]:
    text = make_prompt(prompt_tokens)
    while True:
        yield text, max_tokens


def unique_prompts(prompt_tokens: int, max_tokens: int, seed: int = 0) -> Iterator[tuple[str, int]]:
    """Fixed-length prompts that share no prefix with each other: a per-request nonce first,
    then shuffled filler. Use these when the server has prefix caching on and the experiment
    must pay full prefill every time (fixed_prompts would be served from cache after request 1)."""
    rng = random.Random(seed)
    i = 0
    while True:
        words = [f"req{i}-{rng.randrange(10**9)}"] + [rng.choice(_FILLER) for _ in range(prompt_tokens - 1)]
        yield " ".join(words), max_tokens
        i += 1


def load_sample_rows() -> list[dict]:
    with resources.files("tokenbench.data").joinpath("sample_prompts.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_prompts(seed: int = 0, max_output_tokens: int | None = None) -> Iterator[tuple[str, int]]:
    rows = load_sample_rows()
    random.Random(seed).shuffle(rows)
    i = 0
    while True:
        r = rows[i % len(rows)]
        n = r["output_tokens"] if max_output_tokens is None else min(r["output_tokens"], max_output_tokens)
        yield r["prompt"], n
        i += 1
