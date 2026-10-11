**tokenbench run** · model: Qwen/Qwen2.5-1.5B-Instruct · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 | SLO pass | goodput req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 0 | 66 | 28 ms | 32 ms | 15.1 ms | 15.8 ms | 15.5 ms | 100% | 0.52 |
| 2 | 32 | 0 | 126 | 45 ms | 52 ms | 15.7 ms | 16.4 ms | 16.2 ms | 100% | 0.98 |
| 4 | 32 | 0 | 241 | 50 ms | 65 ms | 16.3 ms | 17.2 ms | 16.8 ms | 100% | 1.88 |
| 8 | 32 | 0 | 440 | 65 ms | 84 ms | 17.7 ms | 19.1 ms | 18.3 ms | 100% | 3.44 |
| 16 | 32 | 0 | 746 | 115 ms | 130 ms | 20.6 ms | 22.4 ms | 21.3 ms | 100% | 5.83 |
| 32 | 32 | 0 | 1058 | 197 ms | 202 ms | 28.7 ms | 31.2 ms | 29.8 ms | 100% | 8.27 |
| 64 | 32 | 0 | 1059 | 202 ms | 206 ms | 28.7 ms | 31.6 ms | 29.7 ms | 100% | 8.27 |

SLO: TTFT ≤ 500 ms and TPOT ≤ 50 ms. Goodput counts only requests meeting both.
