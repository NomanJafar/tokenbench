"""Outputs: JSON (everything), CSV (per-level aggregates), plots, Markdown summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_COLUMNS = [
    "mode", "level", "requests", "errors", "wall_s",
    "requests_per_s", "output_tokens_per_s", "total_tokens_per_s",
    "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_p50_ms", "tpot_p95_ms",
    "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "e2e_p50_s", "e2e_p95_s",
    "slo_pass_rate", "goodput_requests_per_s", "goodput_tokens_per_s",
]


def _ms(x):
    return None if x is None else round(x * 1000, 3)


def level_row(lv: dict) -> dict:
    return {
        "mode": lv["mode"], "level": lv["level"], "requests": lv["requests"], "errors": lv["errors"],
        "wall_s": round(lv["wall_s"], 3),
        "requests_per_s": lv["requests_per_s"], "output_tokens_per_s": lv["output_tokens_per_s"],
        "total_tokens_per_s": lv["total_tokens_per_s"],
        "ttft_p50_ms": _ms(lv["ttft"]["p50"]), "ttft_p95_ms": _ms(lv["ttft"]["p95"]), "ttft_p99_ms": _ms(lv["ttft"]["p99"]),
        "tpot_p50_ms": _ms(lv["tpot"]["p50"]), "tpot_p95_ms": _ms(lv["tpot"]["p95"]),
        "itl_p50_ms": _ms(lv["itl"]["p50"]), "itl_p95_ms": _ms(lv["itl"]["p95"]), "itl_p99_ms": _ms(lv["itl"]["p99"]),
        "e2e_p50_s": lv["e2e"]["p50"], "e2e_p95_s": lv["e2e"]["p95"],
        "slo_pass_rate": lv.get("slo_pass_rate"),
        "goodput_requests_per_s": lv.get("goodput_requests_per_s"),
        "goodput_tokens_per_s": lv.get("goodput_tokens_per_s"),
    }


def write_run(run: dict, json_path: Path | str) -> list[Path]:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(run, indent=1))
    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for lv in run["levels"]:
            w.writerow(level_row(lv))
    return [json_path, csv_path]


def _x_label(run: dict) -> str:
    return "concurrent requests" if run["config"].get("mode") == "closed" else "offered load (requests / s)"


def plot_run(run: dict, out_dir: Path | str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = run["levels"]
    xs = [lv["level"] for lv in levels]
    xlabel = _x_label(run)
    paths = []

    def finish(fig, ax, title, ylabel, name, log_x=True):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_x and min(xs) > 0:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, [lv["output_tokens_per_s"] for lv in levels], marker="o", label="output tokens/s")
    ax.plot(xs, [lv["total_tokens_per_s"] for lv in levels], marker="s", alpha=0.6, label="total tokens/s")
    finish(fig, ax, "Throughput vs load", "tokens / s", "throughput_vs_load.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p_ in ("p50", "p95", "p99"):
        a1.plot(xs, [lv["ttft"][p_] * 1000 for lv in levels], marker="o", label=f"TTFT {p_}")
        a2.plot(xs, [lv["itl"][p_] * 1000 for lv in levels], marker="o", label=f"ITL {p_}")
    a1.set_yscale("log")
    a1.set_ylabel("ms (log)")
    a1.set_title("Time to first token")
    a2.set_ylabel("ms")
    a2.set_title("Inter-token latency")
    for a in (a1, a2):
        a.set_xlabel(xlabel)
        if min(xs) > 0:
            a.set_xscale("log", base=2)
            a.set_xticks(xs)
            a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.grid(True, alpha=0.3)
        a.legend()
    fig.suptitle("Latency percentiles vs load")
    fig.tight_layout()
    p = out_dir / "latency_vs_load.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    last = levels[-1]
    itl = last["itl"]
    # No raw ITL samples are kept per level (too large); show the percentile ladder instead.
    ax.bar(["min", "p50", "p95", "p99", "max"], [itl[k] * 1000 for k in ("min", "p50", "p95", "p99", "max")])
    finish(fig, ax, f"ITL percentile ladder at level {last['level']} (n={itl['n']})", "ms",
           "itl_distribution.png", log_x=False)

    if any(lv.get("slo") for lv in levels):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(xs, [lv.get("requests_per_s") or 0 for lv in levels], marker="o", label="throughput (req/s)")
        ax.plot(xs, [lv.get("goodput_requests_per_s") or 0 for lv in levels], marker="o", label="goodput (req/s meeting SLO)")
        slo = next(lv["slo"] for lv in levels if lv.get("slo"))
        finish(fig, ax, f"Goodput vs load (SLO: TTFT ≤ {slo['ttft_ms']:.0f} ms, TPOT ≤ {slo['tpot_ms']:.0f} ms)",
               "requests / s", "goodput_vs_load.png")
    return paths


def markdown_summary(run: dict) -> str:
    cfg = run["config"]
    has_slo = any(lv.get("slo") for lv in run["levels"])
    head = ["level", "req", "err", "out tok/s", "TTFT p50", "TTFT p95", "ITL p50", "ITL p95", "TPOT p50"]
    if has_slo:
        head += ["SLO pass", "goodput req/s"]
    lines = [
        f"**tokenbench run** · model: {cfg.get('model')} · mode: {cfg.get('mode')} · server: {cfg.get('base_url')}",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    f = lambda x, s=1, d=1: "-" if x is None else f"{x * s:.{d}f}"
    for lv in run["levels"]:
        row = [str(lv["level"]), str(lv["requests"]), str(lv["errors"]), f(lv["output_tokens_per_s"], 1, 0),
               f(lv["ttft"]["p50"], 1000, 0) + " ms", f(lv["ttft"]["p95"], 1000, 0) + " ms",
               f(lv["itl"]["p50"], 1000, 1) + " ms", f(lv["itl"]["p95"], 1000, 1) + " ms",
               f(lv["tpot"]["p50"], 1000, 1) + " ms"]
        if has_slo:
            row += [f(lv.get("slo_pass_rate"), 100, 0) + "%", f(lv.get("goodput_requests_per_s"), 1, 2)]
        lines.append("| " + " | ".join(row) + " |")
    if has_slo:
        slo = next(lv["slo"] for lv in run["levels"] if lv.get("slo"))
        lines += ["", (f"SLO: TTFT ≤ {slo['ttft_ms']:.0f} ms and TPOT ≤ {slo['tpot_ms']:.0f} ms. "
                      "Goodput counts only requests meeting both.")]
    return "\n".join(lines) + "\n"
