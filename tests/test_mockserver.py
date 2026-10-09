import asyncio
import json
import time
from itertools import pairwise

import httpx

from tokenbench.mockserver import MockConfig, create_app


def client(**cfg):
    app = create_app(MockConfig(**cfg))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock")


def body(max_tokens=4, stream=True, **extra):
    return {"model": "mock", "messages": [{"role": "user", "content": "one two three"}],
            "max_tokens": max_tokens, "stream": stream, "ignore_eos": True,
            "stream_options": {"include_usage": True}, **extra}


async def read_stream(c, b):
    chunks, t0 = [], time.monotonic()
    async with c.stream("POST", "/v1/chat/completions", json=b) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                chunks.append((time.monotonic() - t0, json.loads(line[5:])))
    return chunks


def test_non_streaming_completion_reports_usage():
    async def go():
        async with client(ttft_ms=1, itl_ms=1) as c:
            r = await c.post("/v1/chat/completions", json=body(max_tokens=6, stream=False))
            return r.status_code, r.json()
    status, j = asyncio.run(go())
    assert status == 200
    assert j["usage"]["completion_tokens"] == 6
    assert j["usage"]["prompt_tokens"] == 3
    assert j["choices"][0]["message"]["content"].split().__len__() == 6


def test_streaming_emits_exactly_max_tokens_content_chunks_then_usage():
    async def go():
        async with client(ttft_ms=1, itl_ms=1) as c:
            return await read_stream(c, body(max_tokens=5))
    chunks = asyncio.run(go())
    content = [c for _, c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert len(content) == 5
    assert chunks[-1][1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_ttft_and_itl_follow_config_when_under_capacity(live_mock):
    async def go(url):
        async with httpx.AsyncClient(base_url=url) as c:
            return await read_stream(c, body(max_tokens=4))
    with live_mock(ttft_ms=100, itl_ms=20, capacity=4) as url:
        chunks = asyncio.run(go(url))
    content_t = [t for t, c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert 0.09 <= content_t[0] <= 0.2
    gaps = [b - a for a, b in pairwise(content_t)]
    assert all(0.015 <= g <= 0.06 for g in gaps)


def test_requests_beyond_capacity_queue_and_pay_in_ttft(live_mock):
    async def go(url):
        async with httpx.AsyncClient(base_url=url) as c:
            return await asyncio.gather(read_stream(c, body(max_tokens=5)), read_stream(c, body(max_tokens=5)))
    with live_mock(ttft_ms=50, itl_ms=10, capacity=1) as url:
        a, b = asyncio.run(go(url))
    first_tok = lambda ch: next(t for t, c in ch if c["choices"] and c["choices"][0]["delta"].get("content"))
    fast, slow = sorted([first_tok(a), first_tok(b)])
    assert fast < 0.15
    assert slow > fast + 0.08   # waited for the other request's ~90 ms (50 ttft + 4x10 itl) to finish


def test_metrics_expose_running_waiting_and_token_counters():
    async def go():
        async with client(ttft_ms=60, itl_ms=10, capacity=1) as c:
            t1 = asyncio.create_task(read_stream(c, body(max_tokens=3)))
            t2 = asyncio.create_task(read_stream(c, body(max_tokens=3)))
            await asyncio.sleep(0.03)
            during = (await c.get("/metrics")).text
            await asyncio.gather(t1, t2)
            after = (await c.get("/metrics")).text
            return during, after
    during, after = asyncio.run(go())
    assert "vllm:num_requests_running 1" in during
    assert "vllm:num_requests_waiting 1" in during
    assert "vllm:generation_tokens_total 6" in after
    assert "vllm:prompt_tokens_total 6" in after
    assert "vllm:time_to_first_token_seconds_count 2" in after


def test_error_rate_one_fails_every_request():
    async def go():
        async with client(ttft_ms=1, itl_ms=1, error_rate=1.0) as c:
            return (await c.post("/v1/chat/completions", json=body(stream=False))).status_code
    assert asyncio.run(go()) == 500


def test_admin_config_changes_latency_at_runtime(live_mock):
    async def go(url):
        async with httpx.AsyncClient(base_url=url) as c:
            r = await c.post("/admin/config", json={"ttft_ms": 120})
            assert r.json()["ttft_ms"] == 120
            chunks = await read_stream(c, body(max_tokens=2))
            return next(t for t, ch in chunks if ch["choices"] and ch["choices"][0]["delta"].get("content"))
    with live_mock(ttft_ms=1, itl_ms=1) as url:
        assert asyncio.run(go(url)) >= 0.11


def test_health_and_models_endpoints():
    async def go():
        async with client() as c:
            return (await c.get("/health")).status_code, (await c.get("/v1/models")).json()["data"][0]["id"]
    assert asyncio.run(go()) == (200, "mock")
