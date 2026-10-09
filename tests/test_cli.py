import json

from tokenbench.cli import build_parser, main


def test_parser_defaults_for_sweep():
    a = build_parser().parse_args(["sweep", "--base-url", "http://x", "--model", "m", "--out", "r.json"])
    assert a.cmd == "sweep" and a.mode == "closed" and a.levels == "1,2,4,8,16,32"
    assert a.prompt_tokens == 256 and a.max_tokens == 128 and a.slo_ttft_ms is None


def test_sweep_end_to_end_writes_json_csv_plots_and_summary(live_mock, tmp_path):
    out = tmp_path / "run.json"
    with live_mock(ttft_ms=5, itl_ms=1, capacity=64) as url:
        main(["sweep", "--base-url", url, "--model", "mock", "--out", str(out),
              "--levels", "1,2", "--num-requests", "4", "--warmup", "1",
              "--prompt-tokens", "8", "--max-tokens", "3",
              "--slo-ttft-ms", "500", "--slo-tpot-ms", "50", "--plots"])
    run = json.loads(out.read_text())
    assert [lv["level"] for lv in run["levels"]] == [1, 2]
    assert run["config"]["model"] == "mock" and run["config"]["mode"] == "closed"
    assert "tokenbench_version" in run["config"] and "command" in run["config"]
    assert run["levels"][0]["slo_pass_rate"] == 1.0
    assert (tmp_path / "run.csv").exists()
    assert (tmp_path / "run.md").exists()
    assert (tmp_path / "throughput_vs_load.png").exists()
    assert (tmp_path / "goodput_vs_load.png").exists()


def test_sweep_open_loop_uses_rate_levels_and_duration(live_mock, tmp_path):
    out = tmp_path / "open.json"
    with live_mock(ttft_ms=5, itl_ms=1, capacity=64) as url:
        main(["sweep", "--base-url", url, "--model", "mock", "--out", str(out), "--mode", "open",
              "--levels", "20", "--duration", "0.5", "--warmup", "0", "--prompt-tokens", "5", "--max-tokens", "2"])
    run = json.loads(out.read_text())
    assert run["config"]["mode"] == "open"
    assert 5 <= run["levels"][0]["requests"] <= 20


def test_sweep_sample_prompts_option_varies_output_lengths(live_mock, tmp_path):
    out = tmp_path / "s.json"
    with live_mock(ttft_ms=1, itl_ms=0.2, capacity=64) as url:
        main(["sweep", "--base-url", url, "--model", "mock", "--out", str(out),
              "--levels", "4", "--num-requests", "8", "--warmup", "0", "--prompts", "sample", "--max-tokens", "1024"])
    run = json.loads(out.read_text())
    lengths = {r["completion_tokens"] for r in run["levels"][0]["per_request"]}
    assert len(lengths) > 1 and max(lengths) <= 1024


def test_mockserver_subcommand_parses_flags():
    a = build_parser().parse_args(["mockserver", "--port", "9999", "--ttft-ms", "30", "--capacity", "4"])
    assert a.cmd == "mockserver" and a.port == 9999 and a.ttft_ms == 30 and a.capacity == 4
