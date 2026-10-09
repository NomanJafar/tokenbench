import asyncio
import itertools
import statistics

import httpx

from tokenbench.client import stream_request
from tokenbench.load import RunConfig, run_level
from tokenbench.metrics import SLO
from tokenbench.prompts import fixed_prompts, sample_prompts


def test_fixed_prompts_yield_constant_shape():
    src = fixed_prompts(prompt_tokens=50, max_tokens=8)
    p, m = next(src)
    assert 45 <= len(p.split()) <= 55 and m == 8
    assert next(src) == (p, m)


def test_sample_prompts_cycle_through_bundled_file_deterministically():
    a = [next(sample_prompts(seed=1)) for _ in range(1)]
    b = [next(sample_prompts(seed=1)) for _ in range(1)]
    assert a == b
    rows = [next(src) for src in [sample_prompts(seed=2)] for _ in range(50)]
    lengths = {m for _, m in rows}
    assert len(lengths) > 5                     # real variation in output lengths
    assert all(1 <= m <= 2048 for _, m in rows)


def test_stream_request_returns_record_with_token_times(live_mock):
    async def go(url):
        async with httpx.AsyncClient(base_url=url) as c:
            return await stream_request(c, "mock", "a b c", max_tokens=5)
    with live_mock(ttft_ms=30, itl_ms=5) as url:
        r = asyncio.run(go(url))
    assert r.ok and r.completion_tokens == 5 and r.prompt_tokens == 3
    assert len(r.token_times) == 5 and r.ttft_s >= 0.025
    assert r.meta["status"] == 200


def test_stream_request_records_http_errors(live_mock):
    async def go(url):
        async with httpx.AsyncClient(base_url=url) as c:
            return await stream_request(c, "mock", "a", max_tokens=2)
    with live_mock(error_rate=1.0) as url:
        r = asyncio.run(go(url))
    assert not r.ok and "500" in r.error and r.token_times == []


def test_closed_loop_runs_requested_requests_at_fixed_concurrency(live_mock):
    cfg = RunConfig(model="mock", mode="closed", level=4, num_requests=12, warmup_requests=2,
                    prompts=fixed_prompts(10, 4), metrics_interval_s=0.02)
    with live_mock(ttft_ms=10, itl_ms=2, capacity=64) as url:
        out = asyncio.run(run_level(url, cfg))
    assert out["mode"] == "closed" and out["level"] == 4
    assert out["requests"] == 12 and out["errors"] == 0
    assert out["warmup_requests"] == 2
    assert max(s["running"] for s in out["metrics_timeline"]) <= 4
    assert len(out["per_request"]) == 12


def test_open_loop_issues_requests_at_target_rate_regardless_of_latency(live_mock):
    # 40 req/s for 1 s against a server that takes ~0.5 s per request with capacity 2:
    # a closed loop would send ~4 requests; an open loop must still send ~40.
    cfg = RunConfig(model="mock", mode="open", level=40, duration_s=1.0, warmup_requests=0,
                    prompts=fixed_prompts(5, 2), seed=1)
    with live_mock(ttft_ms=400, itl_ms=50, capacity=2) as url:
        out = asyncio.run(run_level(url, cfg))
    assert 28 <= out["requests"] <= 52
    starts = sorted(r["t_start"] for r in out["per_request"])
    assert statistics.mean(b - a for a, b in itertools.pairwise(starts)) < 0.06
    assert out["ttft"]["p95"] > 1.0                 # queueing shows up in TTFT, as it should


def test_run_level_reports_goodput_when_slo_given(live_mock):
    cfg = RunConfig(model="mock", mode="closed", level=2, num_requests=4, warmup_requests=0,
                    prompts=fixed_prompts(5, 3), slo=SLO(ttft_ms=200, tpot_ms=50))
    with live_mock(ttft_ms=10, itl_ms=2) as url:
        out = asyncio.run(run_level(url, cfg))
    assert out["slo_pass_rate"] == 1.0 and out["goodput_requests_per_s"] > 0
