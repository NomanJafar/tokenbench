# Concepts: how to load-test an LLM server without fooling yourself

`tokenbench` is the measurement tool every later project in this portfolio uses. If the tool is wrong, every result built on it is wrong, so this primer is about methodology more than code.

## 1. The four latencies

For a streaming request, four numbers matter, and they answer different questions.

| Metric | Definition | What a user feels |
|---|---|---|
| **TTFT**, time to first token | request sent → first content token received | "is it responding?" |
| **ITL**, inter-token latency | gap between consecutive tokens, one sample per token | smoothness of the stream |
| **TPOT**, time per output token | (E2E − TTFT) / (output tokens − 1), one number per request | average decode speed for that request |
| **E2E**, end-to-end | request sent → stream closed | total wait for a non-streaming client |

TTFT is dominated by queueing plus prefill, so it reacts to load and prompt length. ITL is dominated by the decode step time, so it reacts to batch size. They can move independently: a server can have excellent ITL and terrible TTFT (long queue, fast once admitted) or the reverse. Report both; never collapse them into one "latency".

ITL and TPOT differ in a subtle way. ITL is a distribution with thousands of samples per run and shows stalls (a p99 ITL of 400 ms with a p50 of 15 ms means periodic hiccups, often prefill of other requests interrupting decode; project 8 is about exactly this). TPOT hides stalls in an average. Use ITL percentiles for diagnosis, TPOT for a per-request summary.

## 2. Throughput has units, and they matter

"Tokens per second" is ambiguous. Be explicit:

- **Output tokens/s**: completion tokens generated per second across all requests. The main capacity number for a decode-heavy workload.
- **Total tokens/s**: prompt + completion tokens per second. Flatters servers given long prompts; prefill tokens are cheap per token compared to decode tokens.
- **Requests/s**: only meaningful with fixed request shapes.

Per-request throughput (one user's tokens/s, = 1/ITL) and aggregate throughput (all users) move in opposite directions as load rises. The whole point of a load curve is to show that trade-off.

## 3. Closed loop vs open loop, and coordinated omission

**Closed loop**: N workers; each sends a request, waits for the full response, sends the next. The offered load adapts to the server: when it slows down, workers send less. Good for answering "what does the system look like with N concurrent sessions?" and for mapping the throughput/latency curve by sweeping N. Bad for modelling real traffic, because real users do not politely wait for other users' requests to finish.

**Open loop**: requests arrive on a schedule (typically a Poisson process with rate λ) regardless of whether earlier ones have finished. If the server cannot keep up, the queue grows without bound and latency climbs until the run ends. This is how production traffic behaves, and it is the only way to find the **saturation point**: the arrival rate above which latency stops being bounded.

**Coordinated omission** is the failure mode of closed-loop measurement. When a server stalls for two seconds, a closed-loop client sends no requests during the stall, so the stall shows up as one slow sample instead of the dozens of slow requests real users would have experienced. Latency percentiles come out far too optimistic. Open-loop testing does not have this problem, because requests scheduled during the stall still get sent (and measured as slow).

`tokenbench` implements both. The rule for using them: closed loop for mapping curves against concurrency (projects 1, 5, 6, 7), open loop for anything that claims a latency SLO at a traffic level (projects 8, 13, 14, 15).

## 4. Little's law connects them

For a stable system, `concurrency = arrival_rate × latency` (L = λW). At 5 requests/s and 2 s average E2E latency, about 10 requests are in flight on average. This is useful two ways: to pick a closed-loop N that corresponds to a target open-loop rate, and to sanity-check results. If a sweep reports 20 req/s at 4 s latency with only 8 concurrent workers, something is mislabelled.

## 5. Percentiles, not averages

Latency distributions are long-tailed. The mean is pulled by a few slow requests and describes nobody's experience. Report p50 (typical), p95 (what one in twenty users sees) and p99 (what the loudest complainers see), and say how many samples each is based on: a p99 computed from 50 requests is a single sample and is noise.

Rule of thumb used here: at least 200 samples for p95, at least 1,000 for p99. For ITL this is easy (every token is a sample). For TTFT it dictates run length.

Percentile interpolation method also matters at small n; `tokenbench` uses linear interpolation (NumPy's default) and says so in its output.

## 6. Goodput: throughput that meets the SLO

Raw throughput can be increased almost indefinitely by admitting more requests and letting latency explode. The number that matters for capacity planning is **goodput**: requests (or tokens) per second that *met the latency target*, for example TTFT ≤ 500 ms and ITL p95 ≤ 50 ms. Plotted against offered load, goodput rises with throughput and then collapses when the SLO starts failing. The peak of the goodput curve is the real capacity of the configuration. Project 15 reports this for three configurations.

## 7. Control what you can, record what you cannot

Variation in these things swamps most effects you are trying to measure:

- **Output length.** A 20-token answer and a 500-token answer differ 25× in decode cost. For curve-mapping, fix output length exactly (vLLM honours `ignore_eos` with `max_tokens`). For realistic runs, use a recorded length distribution and report it.
- **Prompt length.** Fix it, or sample from a stated distribution.
- **Sampling.** `temperature` does not change compute cost, but it changes when EOS appears, hence output length. Use a fixed seed or `ignore_eos`.
- **Warm-up.** The first requests after startup pay for CUDA graph capture, allocator warm-up and cold caches. Discard a warm-up phase and say how long it was.
- **Run length.** Short runs over-weight start-up transients. Each measured point should run for at least 30–60 s or a few hundred requests.
- **Client overhead.** Python clients add ~ms of jitter. Record client-side numbers, compare against the server's own `/metrics` histograms, and explain any gap.
- **Everything else.** GPU model, driver, engine version, model revision, dtype, every engine flag, and the exact tokenbench command go into the results file. A result without its configuration is an anecdote.

## 8. Reading a load curve

A healthy throughput-vs-concurrency curve on a memory-bound decode workload rises nearly linearly, then bends into a plateau. The bend ("knee") is where one of three things happened:

1. Decode stopped being memory-bound and became compute-bound (ITL starts rising linearly with batch size).
2. The KV cache filled: requests queue (`num_requests_waiting` > 0, TTFT jumps) or get preempted.
3. The scheduler cap (`max-num-seqs`) was hit, which looks like case 2 but is a configuration choice.

The `/metrics` timeline recorded alongside each run tells you which. A good write-up names the cause of the knee, not just its location.

## 9. Where the input prompts come from

Synthetic filler prompts are fine for fixing length. For realistic token distributions, this tool ships one small sample file of chat-style prompts with recorded output lengths (ShareGPT-style, a few hundred rows) and a loader for it. Anything bigger is out of scope; the loader interface is what matters.

## 10. The mock server

Building gateways, autoscalers and chaos tooling against a real GPU is slow and expensive. `tokenbench.mockserver` is an OpenAI-compatible fake that streams tokens with configurable TTFT, ITL, a concurrency cap (requests beyond it queue, so TTFT grows the way it does on a saturated vLLM), and injectable faults (latency spikes, error rates, hard stalls, process death). It exposes a `/metrics` endpoint with the same series `servellm.metrics` watches. Its job is to make the *control* logic in projects 4, 11, 12, 13 and 14 testable on a laptop in seconds; the final validation of each still happens once on a T4.

## 11. What tokenbench produces

- A `results/<run>.json` with: full configuration, per-request records (TTFT, TPOT, ITLs, tokens, errors, timestamps), per-level aggregates (p50/p95/p99 for each latency, throughput in all three units, goodput for the configured SLO), and the `/metrics` timeline if the server exposes one.
- A CSV of per-level aggregates for spreadsheets.
- Plots: throughput vs load, TTFT and ITL percentiles vs load, goodput vs load, ITL distribution.
- A one-page Markdown summary ready to paste into a README.

## Further reading

- Gil Tene, *How NOT to Measure Latency* (talk): the canonical explanation of coordinated omission.
- Zhong et al., *DistServe* (OSDI 2024): introduces goodput as the metric for LLM serving and motivates prefill/decode disaggregation (project 10).
- vLLM's own `benchmark_serving.py` and the `GenAI-Perf` / `LLMPerf` tools: read their option lists to see what practitioners decided was worth controlling.
