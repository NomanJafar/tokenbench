"""Load generators: closed loop (N workers) and open loop (Poisson arrivals).

See CONCEPTS.md section 3 for why both exist and when to use which.
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from tokenbench.client import stream_request
from tokenbench.metrics import SLO, RequestRecord, aggregate
from tokenbench.prompts import fixed_prompts

WATCHED = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total",),
}


@dataclass
class RunConfig:
    model: str
    mode: str = "closed"                 # "closed": level = concurrency; "open": level = requests/s
    level: float = 1
    num_requests: int | None = None      # closed loop: total requests (after warm-up)
    duration_s: float | None = None      # open loop: how long to keep issuing requests
    warmup_requests: int = 4
    prompts: Iterator[tuple[str, int]] = field(default_factory=lambda: fixed_prompts(256, 128))
    ignore_eos: bool = True
    slo: SLO | None = None
    metrics_interval_s: float = 0.5
    seed: int = 0
    system_prompt: str | None = None
    extra_headers: dict | None = None
    drain_timeout_s: float = 120.0
    request_timeout_s: float | None = 300.0   # a wedged server must not hang a closed loop forever
    max_connections: int | None = 2000        # httpx defaults to 100, which caps open-loop concurrency silently


_PROM_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name -> value; labels dropped, same-name series summed."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM_LINE.match(line)
        if m and not line.startswith("#"):
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                continue
    return out


async def _sample_metrics(client: httpx.AsyncClient, interval_s: float, out: list[dict], stop: asyncio.Event, t0: float):
    while not stop.is_set():
        try:
            resp = await client.get("/metrics", timeout=5.0)
            if resp.status_code == 200:
                raw = parse_prometheus(resp.text)
                snap = {"t": time.monotonic() - t0}
                for short, names in WATCHED.items():
                    for n in names:
                        if n in raw:
                            snap[short] = raw[n]
                            break
                out.append(snap)
        except httpx.HTTPError:
            pass
        try:
            await asyncio.wait_for(stop.wait(), interval_s)
        except TimeoutError:
            pass


async def _one(client: httpx.AsyncClient, cfg: RunConfig, prompt: str, max_tokens: int) -> RequestRecord:
    return await stream_request(client, cfg.model, prompt, max_tokens, ignore_eos=cfg.ignore_eos,
                                seed=cfg.seed, system=cfg.system_prompt, extra_headers=cfg.extra_headers,
                                request_timeout_s=cfg.request_timeout_s)


async def _closed(client: httpx.AsyncClient, cfg: RunConfig, records: list[RequestRecord]) -> int | None:
    """Returns the original num_requests if it had to be raised, else None."""
    assert cfg.num_requests, "closed loop needs num_requests"
    adjusted_from = None
    n = cfg.num_requests
    if n < cfg.level:
        # Fewer requests than workers measures a smaller concurrency than claimed.
        adjusted_from, n = n, int(cfg.level) * 2
        print(f"closed loop: num_requests {adjusted_from} < concurrency {int(cfg.level)}; raised to {n}", file=sys.stderr)
    remaining = n

    async def worker():
        nonlocal remaining
        while remaining > 0:
            remaining -= 1
            prompt, n = next(cfg.prompts)
            records.append(await _one(client, cfg, prompt, n))

    await asyncio.gather(*[worker() for _ in range(int(cfg.level))])
    return adjusted_from


async def _open(client: httpx.AsyncClient, cfg: RunConfig, records: list[RequestRecord]) -> None:
    assert cfg.duration_s, "open loop needs duration_s"
    rng = random.Random(cfg.seed)
    tasks: list[asyncio.Task] = []
    t_end = time.monotonic() + cfg.duration_s
    next_at = time.monotonic()
    while next_at < t_end:
        await asyncio.sleep(max(0.0, next_at - time.monotonic()))
        prompt, n = next(cfg.prompts)
        tasks.append(asyncio.create_task(_one(client, cfg, prompt, n)))
        next_at += rng.expovariate(cfg.level)  # Poisson process: exponential inter-arrival times
    done, pending = await asyncio.wait(tasks, timeout=cfg.drain_timeout_s)
    for t in pending:
        t.cancel()
    records.extend(t.result() for t in done)
    records.extend(RequestRecord(t_start=0.0, token_times=[], t_end=0.0, prompt_tokens=None,
                                 completion_tokens=0, error="drain timeout") for _ in pending)


def make_client(base_url: str, cfg: RunConfig) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=cfg.max_connections, max_keepalive_connections=cfg.max_connections)
    return httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits)


async def run_level(base_url: str, cfg: RunConfig) -> dict:
    async with make_client(base_url, cfg) as client:
        for _ in range(cfg.warmup_requests):
            prompt, n = next(cfg.prompts)
            await _one(client, cfg, prompt, n)

        records: list[RequestRecord] = []
        timeline: list[dict] = []
        stop = asyncio.Event()
        t0 = time.monotonic()
        sampler = asyncio.create_task(_sample_metrics(client, cfg.metrics_interval_s, timeline, stop, t0))
        adjusted_from = None
        if cfg.mode == "closed":
            adjusted_from = await _closed(client, cfg, records)
        elif cfg.mode == "open":
            await _open(client, cfg, records)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        wall = time.monotonic() - t0
        stop.set()
        await sampler

    out = {"mode": cfg.mode, "level": cfg.level, "warmup_requests": cfg.warmup_requests,
           **aggregate(records, wall, cfg.slo)}
    out["num_requests_adjusted_from"] = adjusted_from
    out["metrics_timeline"] = timeline
    out["per_request"] = [
        {"t_start": r.t_start - t0, "ttft_s": r.ttft_s, "tpot_s": r.tpot_s, "e2e_s": r.e2e_s,
         "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "error": r.error}
        for r in records
    ]
    return out
