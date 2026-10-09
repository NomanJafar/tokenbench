import pytest

from tokenbench.metrics import SLO, RequestRecord, aggregate, latency_summary, percentile


def rec(t0=0.0, ttft=0.1, itl=0.02, n=5, prompt=10, error=None):
    """A request whose first token lands at t0+ttft and then one token every `itl`."""
    times = [t0 + ttft + i * itl for i in range(n)]
    return RequestRecord(t_start=t0, token_times=times, t_end=times[-1] if times else t0,
                         prompt_tokens=prompt, completion_tokens=n, error=error)


def test_request_record_derives_the_four_latencies():
    r = rec(t0=1.0, ttft=0.25, itl=0.05, n=4)
    assert r.ttft_s == pytest.approx(0.25)
    assert r.itls_s == pytest.approx([0.05, 0.05, 0.05])
    assert r.tpot_s == pytest.approx(0.05)           # (e2e - ttft) / (n - 1)
    assert r.e2e_s == pytest.approx(0.25 + 0.15)


def test_request_record_with_no_tokens_has_none_latencies():
    r = RequestRecord(t_start=0.0, token_times=[], t_end=0.5, prompt_tokens=3, completion_tokens=0, error="timeout")
    assert r.ttft_s is None and r.tpot_s is None and r.itls_s == []


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 50) is None


def test_latency_summary_reports_percentiles_and_sample_count():
    s = latency_summary([0.1, 0.2, 0.3, 0.4])
    assert s["n"] == 4 and s["p50"] == pytest.approx(0.25) and s["mean"] == pytest.approx(0.25)
    assert set(s) == {"n", "mean", "p50", "p95", "p99", "min", "max"}
    assert latency_summary([])["n"] == 0 and latency_summary([])["p50"] is None


def test_aggregate_throughput_in_three_units():
    rs = [rec(n=10, prompt=20), rec(n=10, prompt=20)]
    a = aggregate(rs, wall_s=4.0)
    assert a["requests"] == 2 and a["errors"] == 0
    assert a["output_tokens_per_s"] == pytest.approx(5.0)
    assert a["total_tokens_per_s"] == pytest.approx(15.0)
    assert a["requests_per_s"] == pytest.approx(0.5)
    assert a["ttft"]["p50"] == pytest.approx(0.1)
    assert a["itl"]["n"] == 18                        # pooled ITL samples: 9 per request


def test_aggregate_goodput_counts_only_requests_meeting_slo():
    good = rec(ttft=0.1, itl=0.02, n=10)
    slow_ttft = rec(ttft=0.9, itl=0.02, n=10)
    slow_decode = rec(ttft=0.1, itl=0.08, n=10)
    failed = rec(n=0, error="500")
    a = aggregate([good, slow_ttft, slow_decode, failed], wall_s=2.0, slo=SLO(ttft_ms=500, tpot_ms=50))
    assert a["errors"] == 1
    assert a["slo"] == {"ttft_ms": 500, "tpot_ms": 50}
    assert a["slo_pass_rate"] == pytest.approx(1 / 4)
    assert a["goodput_requests_per_s"] == pytest.approx(0.5)
    assert a["goodput_tokens_per_s"] == pytest.approx(5.0)


def test_aggregate_without_slo_omits_goodput():
    a = aggregate([rec()], wall_s=1.0)
    assert "goodput_requests_per_s" not in a and a["slo"] is None
