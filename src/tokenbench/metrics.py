"""Latency and throughput metrics for LLM load tests. See CONCEPTS.md sections 1-6.

Everything here is pure: records in, numbers out. The load generators produce
`RequestRecord`s; `aggregate` turns a list of them into the per-level summary that
the reports and plots consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np


@dataclass
class RequestRecord:
    t_start: float                      # monotonic seconds, request sent
    token_times: list[float]            # arrival time of each content-bearing chunk
    t_end: float                        # stream closed (or failed)
    prompt_tokens: int | None
    completion_tokens: int
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ttft_s(self) -> float | None:
        return self.token_times[0] - self.t_start if self.token_times else None

    @property
    def itls_s(self) -> list[float]:
        return [b - a for a, b in pairwise(self.token_times)]

    @property
    def e2e_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def tpot_s(self) -> float | None:
        """Mean decode time per output token after the first; None if < 2 tokens."""
        if len(self.token_times) < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.token_times)


@dataclass(frozen=True)
class SLO:
    ttft_ms: float
    tpot_ms: float

    def met_by(self, r: RequestRecord) -> bool:
        if not r.ok or r.ttft_s is None:
            return False
        if r.ttft_s * 1000 > self.ttft_ms:
            return False
        tpot = r.tpot_s
        return tpot is None or tpot * 1000 <= self.tpot_ms


def percentile(xs: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (NumPy default), None for empty input."""
    if not xs:
        return None
    return float(np.percentile(xs, p))


def latency_summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    arr = np.asarray(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate(records: list[RequestRecord], wall_s: float, slo: SLO | None = None) -> dict:
    ok = [r for r in records if r.ok]
    completion = sum(r.completion_tokens for r in ok)
    prompt = sum(r.prompt_tokens or 0 for r in ok)
    per_s = (lambda x: x / wall_s) if wall_s > 0 else (lambda x: None)
    out = {
        "requests": len(records),
        "errors": sum(1 for r in records if not r.ok),
        "wall_s": wall_s,
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "requests_per_s": per_s(len(ok)),
        "output_tokens_per_s": per_s(completion),
        "total_tokens_per_s": per_s(completion + prompt),
        "ttft": latency_summary([r.ttft_s for r in ok if r.ttft_s is not None]),
        "tpot": latency_summary([r.tpot_s for r in ok if r.tpot_s is not None]),
        "itl": latency_summary([itl for r in ok for itl in r.itls_s]),
        "e2e": latency_summary([r.e2e_s for r in ok]),
        "slo": None,
    }
    if slo is not None:
        good = [r for r in records if slo.met_by(r)]
        out["slo"] = {"ttft_ms": slo.ttft_ms, "tpot_ms": slo.tpot_ms}
        out["slo_pass_rate"] = len(good) / len(records) if records else None
        out["goodput_requests_per_s"] = per_s(len(good))
        out["goodput_tokens_per_s"] = per_s(sum(r.completion_tokens for r in good))
    return out
