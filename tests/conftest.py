"""Shared fixtures. `live_mock` runs the mock server in a real uvicorn thread because
httpx's in-process ASGITransport buffers streamed bodies, which hides all timing."""

import contextlib
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from tokenbench.mockserver import MockConfig, create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def run_mock(**cfg):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(MockConfig(**cfg)), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.02)
    else:
        raise RuntimeError("mock server did not start")
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def live_mock():
    """Usage: `with live_mock(ttft_ms=50) as url:`"""
    return run_mock
