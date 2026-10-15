"""Command line: `tokenbench sweep ...` and `tokenbench mockserver ...`."""

from __future__ import annotations

import argparse
import asyncio
import platform
import shlex
import sys
import time
from pathlib import Path

import tokenbench
from tokenbench.load import RunConfig, run_level
from tokenbench.metrics import SLO
from tokenbench.prompts import fixed_prompts, sample_prompts
from tokenbench.report import markdown_summary, plot_run, write_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tokenbench", description="Load-test an OpenAI-compatible LLM server")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run a sweep over concurrency (closed loop) or request rate (open loop)")
    s.add_argument("--base-url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True, type=Path, help="results JSON; CSV/MD/plots go next to it")
    s.add_argument("--mode", choices=["closed", "open"], default="closed")
    s.add_argument("--levels", default="1,2,4,8,16,32",
                   help="closed: concurrency levels; open: request rates per second")
    s.add_argument("--num-requests", type=int, default=None,
                   help="closed loop: requests per level (default: 4 x concurrency, min 16)")
    s.add_argument("--duration", type=float, default=30.0, help="open loop: seconds per level")
    s.add_argument("--warmup", type=int, default=4, help="warm-up requests per level, discarded")
    s.add_argument("--prompts", choices=["fixed", "sample"], default="fixed")
    s.add_argument("--prompt-tokens", type=int, default=256)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--no-ignore-eos", action="store_true", help="let the model stop naturally")
    s.add_argument("--system-prompt", default=None)
    s.add_argument("--header", action="append", default=[], help="extra header, k:v (repeatable)")
    s.add_argument("--slo-ttft-ms", type=float, default=None)
    s.add_argument("--slo-tpot-ms", type=float, default=None)
    s.add_argument("--metrics-interval", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--request-timeout", type=float, default=300.0, help="seconds per request before it counts as an error")
    s.add_argument("--plots", action="store_true", help="also write PNG plots (needs matplotlib)")

    m = sub.add_parser("mockserver", help="run the OpenAI-compatible mock server")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8001)
    m.add_argument("--model", default="mock")
    m.add_argument("--ttft-ms", type=float, default=80)
    m.add_argument("--itl-ms", type=float, default=12)
    m.add_argument("--capacity", type=int, default=32)
    m.add_argument("--error-rate", type=float, default=0.0)
    m.add_argument("--jitter-ms", type=float, default=0.0)
    return p


def _slo(a) -> SLO | None:
    if a.slo_ttft_ms is None and a.slo_tpot_ms is None:
        return None
    return SLO(ttft_ms=a.slo_ttft_ms if a.slo_ttft_ms is not None else float("inf"),
               tpot_ms=a.slo_tpot_ms if a.slo_tpot_ms is not None else float("inf"))


def _prompts(a):
    if a.prompts == "sample":
        return sample_prompts(seed=a.seed, max_output_tokens=a.max_tokens)
    return fixed_prompts(a.prompt_tokens, a.max_tokens)


def run_sweep(a, argv: list[str]) -> dict:
    levels = [float(x) if a.mode == "open" else int(x) for x in a.levels.split(",")]
    headers = dict(h.split(":", 1) for h in a.header) or None
    run = {
        "config": {
            "base_url": a.base_url, "model": a.model, "mode": a.mode, "levels": levels,
            "prompts": a.prompts, "prompt_tokens": a.prompt_tokens, "max_tokens": a.max_tokens,
            "ignore_eos": not a.no_ignore_eos, "warmup": a.warmup, "seed": a.seed,
            "num_requests": a.num_requests, "duration_s": a.duration,
            "slo": {"ttft_ms": a.slo_ttft_ms, "tpot_ms": a.slo_tpot_ms},
            "tokenbench_version": tokenbench.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": "tokenbench " + shlex.join(argv),
        },
        "levels": [],
    }
    for lv in levels:
        cfg = RunConfig(
            model=a.model, mode=a.mode, level=lv,
            num_requests=a.num_requests or max(16, 4 * int(lv)) if a.mode == "closed" else None,
            duration_s=a.duration if a.mode == "open" else None,
            warmup_requests=a.warmup, prompts=_prompts(a), ignore_eos=not a.no_ignore_eos,
            slo=_slo(a), metrics_interval_s=a.metrics_interval, seed=a.seed,
            system_prompt=a.system_prompt, extra_headers=headers, request_timeout_s=a.request_timeout,
        )
        res = asyncio.run(run_level(a.base_url, cfg))
        run["levels"].append(res)
        unit = "conc" if a.mode == "closed" else "req/s"
        tt, it = res["ttft"]["p50"], res["itl"]["p50"]
        print(f"{unit}={lv:<6} req={res['requests']:<4} err={res['errors']:<3} "
              f"out_tok/s={res['output_tokens_per_s'] or 0:7.1f}  "
              f"ttft_p50={(tt or 0) * 1000:6.0f}ms  itl_p50={(it or 0) * 1000:5.1f}ms"
              + (f"  goodput={res['goodput_requests_per_s']:.2f} req/s" if res.get("slo") else ""),
              file=sys.stderr)
        write_run(run, a.out)  # checkpoint after every level: free-tier sessions die
    md = markdown_summary(run)
    a.out.with_suffix(".md").write_text(md)
    if a.plots:
        plot_run(run, a.out.parent)
    print(md, file=sys.stderr)
    return run


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(argv)
    if a.cmd == "sweep":
        run_sweep(a, argv)
    elif a.cmd == "mockserver":
        from tokenbench.mockserver import main as mock_main
        mock_main([f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items() if k != "cmd"])


if __name__ == "__main__":
    main()
