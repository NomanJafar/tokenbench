# tokenbench

Load-testing harness for LLM inference servers: TTFT, inter-token latency and throughput under rising concurrency, with reproducible latency curves.

Project 2 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

A load-testing harness that measures time-to-first-token, inter-token latency and throughput as concurrency rises, and plots the curves.

## Why it matters

Latency claims without load curves are marketing, not engineering.

## Status

Concept primer written (`CONCEPTS.md`). Implementation starts next; results land in `results/` once the GPU runs are done.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/tokenbench/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
