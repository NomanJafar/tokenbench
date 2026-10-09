import csv
import json

from tokenbench.report import markdown_summary, plot_run, write_run


def _level(level, tps, ttft, itl, goodput=None):
    d = {"mode": "closed", "level": level, "requests": 20, "errors": 0, "wall_s": 5.0,
         "output_tokens_per_s": tps, "total_tokens_per_s": tps * 3, "requests_per_s": tps / 100,
         "ttft": {"n": 20, "mean": ttft, "p50": ttft, "p95": ttft * 2, "p99": ttft * 3, "min": ttft, "max": ttft * 3},
         "tpot": {"n": 20, "mean": itl, "p50": itl, "p95": itl * 1.5, "p99": itl * 2, "min": itl, "max": itl * 2},
         "itl": {"n": 2000, "mean": itl, "p50": itl, "p95": itl * 1.5, "p99": itl * 2, "min": itl, "max": itl * 2},
         "e2e": {"n": 20, "mean": 2.0, "p50": 2.0, "p95": 3.0, "p99": 4.0, "min": 1.0, "max": 4.0},
         "slo": None, "metrics_timeline": [{"t": 0.0, "running": level}],
         "per_request": [{"t_start": 0.0, "ttft_s": ttft, "tpot_s": itl, "e2e_s": 2.0,
                          "prompt_tokens": 10, "completion_tokens": 100, "error": None}] * 2}
    if goodput is not None:
        d["slo"] = {"ttft_ms": 500, "tpot_ms": 50}
        d["goodput_requests_per_s"] = goodput
        d["goodput_tokens_per_s"] = goodput * 100
        d["slo_pass_rate"] = 0.9
    return d


def _run(goodput=False):
    return {"config": {"base_url": "http://x", "model": "m", "mode": "closed", "tokenbench_version": "0.1.0"},
            "levels": [_level(1, 90, 0.05, 0.011, 0.8 if goodput else None),
                       _level(8, 600, 0.08, 0.013, 5.5 if goodput else None),
                       _level(32, 1500, 0.3, 0.02, 2.0 if goodput else None)]}


def test_write_run_produces_json_and_csv(tmp_path):
    paths = write_run(_run(), tmp_path / "run.json")
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["levels"][1]["output_tokens_per_s"] == 600
    with open(tmp_path / "run.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["level"] for r in rows] == ["1", "8", "32"]
    assert float(rows[2]["ttft_p95_ms"]) == 600.0
    assert set(paths) == {tmp_path / "run.json", tmp_path / "run.csv"}


def test_plot_run_writes_expected_figures(tmp_path):
    paths = plot_run(_run(goodput=True), tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["goodput_vs_load.png", "itl_distribution.png", "latency_vs_load.png", "throughput_vs_load.png"]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_plot_run_skips_goodput_without_slo(tmp_path):
    names = {p.name for p in plot_run(_run(), tmp_path)}
    assert "goodput_vs_load.png" not in names


def test_markdown_summary_has_table_with_one_row_per_level():
    md = markdown_summary(_run(goodput=True))
    assert md.count("\n| 1 |") + md.count("\n| 8 |") + md.count("\n| 32 |") == 3
    assert "goodput" in md.lower() and "p95" in md
    assert "model: m" in md
