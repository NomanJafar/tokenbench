"""Streaming OpenAI-compatible client that timestamps every token."""

from __future__ import annotations

import json
import time

import httpx

from tokenbench.metrics import RequestRecord


def parse_sse_line(line: str) -> dict | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


async def stream_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool = True,
    temperature: float = 0.0,
    seed: int | None = 0,
    system: str | None = None,
    extra_headers: dict | None = None,
) -> RequestRecord:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body: dict = {
        "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    if ignore_eos:
        body["ignore_eos"] = True  # vLLM extension: force exactly max_tokens output tokens

    token_times: list[float] = []
    usage: dict | None = None
    status = None
    headers: dict = {}
    error = None
    n_chunks = 0
    t_start = time.monotonic()
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=extra_headers, timeout=None) as resp:
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status != 200:
                error = f"HTTP {status}: {(await resp.aread())[:200].decode(errors='replace')}"
            else:
                async for line in resp.aiter_lines():
                    obj = parse_sse_line(line)
                    if obj is None:
                        continue
                    now = time.monotonic()
                    if any((ch.get("delta") or {}).get("content") for ch in obj.get("choices", [])):
                        token_times.append(now)
                        n_chunks += 1
                    if obj.get("usage"):
                        usage = obj["usage"]
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    t_end = time.monotonic()
    return RequestRecord(
        t_start=t_start, token_times=token_times, t_end=t_end,
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
        completion_tokens=usage.get("completion_tokens") if usage else n_chunks,
        error=error,
        meta={"status": status, "max_tokens": max_tokens, "chunks": n_chunks, "headers": headers},
    )
