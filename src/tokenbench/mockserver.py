"""An OpenAI-compatible fake LLM server with controllable latency, capacity and faults.

Why it exists: gateways, routers, autoscalers and chaos tooling (projects 4, 11-14)
are control logic. Developing them against a real GPU is slow and expensive. This
server reproduces the *shape* of a vLLM replica's behaviour - a TTFT, a steady
inter-token latency, a concurrency cap beyond which requests queue and TTFT grows,
and a vLLM-style /metrics endpoint - in a process that starts in milliseconds.

It is not a simulator of GPU internals. ITL does not rise with batch size here;
if a project needs that, it measures a real server.

Run: `tokenbench mockserver --ttft-ms 80 --itl-ms 12 --capacity 32 --port 8001`
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse


@dataclass
class MockConfig:
    model: str = "mock"
    ttft_ms: float = 80.0            # prefill + scheduling delay before the first token
    itl_ms: float = 12.0             # steady inter-token latency
    capacity: int = 32               # concurrent streams; beyond this, requests queue
    natural_output_tokens: int = 64  # output length when the client does not set ignore_eos
    error_rate: float = 0.0          # fraction of requests failed with HTTP 500
    extra_latency_ms: float = 0.0    # added to TTFT; chaos tooling bumps this to emulate throttling
    jitter_ms: float = 0.0           # uniform +/- jitter on each inter-token gap
    seed: int | None = 0


class _State:
    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.capacity)
        self.running = 0
        self.waiting = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.tpot_sum = 0.0
        self.tpot_count = 0
        self.requests_total = 0
        self.errors_total = 0
        self.rng = random.Random(cfg.seed)


def _count_prompt_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


def create_app(cfg: MockConfig | None = None) -> FastAPI:
    cfg = cfg or MockConfig()
    st = _State(cfg)
    app = FastAPI(title="tokenbench mockserver")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": st.cfg.model, "object": "model", "owned_by": "tokenbench"}]}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        c = st
        return "\n".join([
            f"vllm:num_requests_running {c.running}",
            f"vllm:num_requests_waiting {c.waiting}",
            f"vllm:kv_cache_usage_perc {min(1.0, c.running / max(1, c.cfg.capacity)):.4f}",
            f"vllm:prompt_tokens_total {c.prompt_tokens_total}",
            f"vllm:generation_tokens_total {c.generation_tokens_total}",
            f"vllm:time_to_first_token_seconds_sum {c.ttft_sum:.6f}",
            f"vllm:time_to_first_token_seconds_count {c.ttft_count}",
            f"vllm:time_per_output_token_seconds_sum {c.tpot_sum:.6f}",
            f"vllm:time_per_output_token_seconds_count {c.tpot_count}",
            f"mock:requests_total {c.requests_total}",
            f"mock:errors_total {c.errors_total}",
            "",
        ])

    @app.get("/admin/config")
    async def get_config():
        return asdict(st.cfg)

    @app.post("/admin/config")
    async def set_config(request: Request):
        patch = await request.json()
        for k, v in patch.items():
            if not hasattr(st.cfg, k):
                raise HTTPException(400, f"unknown field {k}")
            setattr(st.cfg, k, v)
        if "capacity" in patch:
            st.sem = asyncio.Semaphore(st.cfg.capacity)
        return asdict(st.cfg)

    @app.post("/admin/kill")
    async def kill():
        # Hard exit for chaos experiments: emulates a replica process dying mid-load.
        os._exit(137)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        st.requests_total += 1
        if st.cfg.error_rate and st.rng.random() < st.cfg.error_rate:
            st.errors_total += 1
            raise HTTPException(500, "injected failure")

        n_prompt = _count_prompt_tokens(body.get("messages", []))
        max_tokens = int(body.get("max_tokens") or st.cfg.natural_output_tokens)
        n_out = max_tokens if body.get("ignore_eos") else min(max_tokens, st.cfg.natural_output_tokens)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": st.cfg.model,
                   "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                obj["usage"] = usage
            return f"data: {json.dumps(obj)}\n\n".encode()

        def usage() -> dict:
            return {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}

        async def gap() -> None:
            j = st.cfg.jitter_ms
            ms = st.cfg.itl_ms + (st.rng.uniform(-j, j) if j else 0.0)
            await asyncio.sleep(max(0.0, ms) / 1000)

        async def generate():
            st.waiting += 1
            t_req = time.monotonic()
            async with st.sem:
                st.waiting -= 1
                st.running += 1
                try:
                    yield chunk({"role": "assistant", "content": ""})
                    await asyncio.sleep((st.cfg.ttft_ms + st.cfg.extra_latency_ms) / 1000)
                    t_first = time.monotonic()
                    st.ttft_sum += t_first - t_req
                    st.ttft_count += 1
                    for i in range(n_out):
                        if i:
                            await gap()
                        yield chunk({"content": f"tok{i} "})
                        st.generation_tokens_total += 1
                    if n_out > 1:
                        st.tpot_sum += (time.monotonic() - t_first) / (n_out - 1)
                        st.tpot_count += 1
                    st.prompt_tokens_total += n_prompt
                    yield chunk({}, finish="stop")
                    if include_usage:
                        yield chunk({}, usage=usage())
                    yield b"data: [DONE]\n\n"
                finally:
                    st.running -= 1

        if body.get("stream"):
            return StreamingResponse(generate(), media_type="text/event-stream")

        # Non-streaming: same timing, one JSON body at the end.
        text = []
        async for raw in generate():
            if raw.startswith(b"data: {"):
                obj = json.loads(raw[6:])
                for ch in obj.get("choices", []):
                    text.append(ch["delta"].get("content") or "")
        return {"id": rid, "object": "chat.completion", "created": created, "model": st.cfg.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text).strip()},
                             "finish_reason": "stop"}],
                "usage": usage()}

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="OpenAI-compatible mock LLM server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="mock")
    p.add_argument("--ttft-ms", type=float, default=80)
    p.add_argument("--itl-ms", type=float, default=12)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--error-rate", type=float, default=0.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    a = p.parse_args(argv)
    cfg = MockConfig(model=a.model, ttft_ms=a.ttft_ms, itl_ms=a.itl_ms, capacity=a.capacity,
                     error_rate=a.error_rate, jitter_ms=a.jitter_ms)
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
