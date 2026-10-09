**tokenbench run** · model: mock · mode: closed · server: http://127.0.0.1:8765

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 | SLO pass | goodput req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 0 | 65 | 64 ms | 64 ms | 13.8 ms | 15.8 ms | 13.8 ms | 100% | 2.03 |
| 2 | 32 | 0 | 134 | 63 ms | 66 ms | 13.4 ms | 15.2 ms | 13.3 ms | 100% | 4.20 |
| 4 | 32 | 0 | 275 | 63 ms | 67 ms | 12.9 ms | 14.6 ms | 12.9 ms | 100% | 8.58 |
| 8 | 32 | 0 | 556 | 64 ms | 69 ms | 12.7 ms | 14.3 ms | 12.7 ms | 100% | 17.37 |
| 16 | 32 | 0 | 1113 | 68 ms | 70 ms | 12.4 ms | 14.0 ms | 12.3 ms | 100% | 34.78 |
| 32 | 32 | 0 | 1103 | 292 ms | 526 ms | 12.2 ms | 14.1 ms | 12.2 ms | 50% | 17.23 |
| 64 | 32 | 0 | 1116 | 291 ms | 520 ms | 12.3 ms | 14.1 ms | 12.3 ms | 50% | 17.44 |

SLO: TTFT ≤ 500 ms and TPOT ≤ 20 ms. Goodput counts only requests meeting both.
