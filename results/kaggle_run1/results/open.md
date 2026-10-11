**tokenbench run** · model: Qwen/Qwen2.5-1.5B-Instruct · mode: open · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 | SLO pass | goodput req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 19 | 0 | 76 | 46 ms | 63 ms | 15.4 ms | 16.2 ms | 15.5 ms | 100% | 0.41 |
| 1.0 | 37 | 0 | 139 | 50 ms | 61 ms | 15.7 ms | 16.5 ms | 15.8 ms | 100% | 0.77 |
| 2.0 | 81 | 0 | 296 | 53 ms | 75 ms | 16.6 ms | 18.4 ms | 16.9 ms | 100% | 1.69 |
| 4.0 | 166 | 0 | 581 | 60 ms | 90 ms | 18.7 ms | 24.9 ms | 19.2 ms | 100% | 3.36 |
| 8.0 | 347 | 0 | 1133 | 84 ms | 109 ms | 27.4 ms | 33.9 ms | 27.5 ms | 100% | 6.92 |
| 16.0 | 716 | 0 | 2013 | 1833 ms | 5305 ms | 42.7 ms | 49.4 ms | 43.1 ms | 29% | 3.60 |

SLO: TTFT ≤ 1000 ms and TPOT ≤ 50 ms. Goodput counts only requests meeting both.
