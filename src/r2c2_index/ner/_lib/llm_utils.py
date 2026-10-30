"""Shared LLM utilities for demo scripts.

Provides EndpointPool (round-robin async OpenAI client pool with proper
httpx connection limits) used by extraction, refinement, and judge scripts.
"""


class EndpointPool:
    """Round-robin pool of AsyncOpenAI clients.

    No client-side concurrency limits — vLLM's continuous batching scheduler
    handles queuing internally. The httpx connection pool is set high to
    avoid becoming the bottleneck when many coroutines run concurrently.
    """

    def __init__(self, urls: list[str]) -> None:
        import httpx
        from openai import AsyncOpenAI

        # Default httpx pool is 100 connections — far too low for thousands of
        # concurrent requests. Set high and let vLLM handle scheduling.
        pool_limits = httpx.Limits(
            max_connections=10000,
            max_keepalive_connections=1000,
        )

        self._clients: list = []
        for url in urls:
            http_client = httpx.AsyncClient(limits=pool_limits)
            self._clients.append(
                AsyncOpenAI(base_url=url, api_key="unused", http_client=http_client)
            )
        self._counter = 0
        self._n = len(urls)

    def next(self):
        """Return the next client (round-robin)."""
        idx = self._counter % self._n
        self._counter += 1
        return self._clients[idx]
