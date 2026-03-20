"""backend/model_client.py — Async client for the model server (streaming inference)."""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8001")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
REQUEST_TIMEOUT = float(os.environ.get("MODEL_REQUEST_TIMEOUT", 300))


class ModelClient:
    """
    Async HTTP client that talks to the model server.
    Supports both streaming (SSE) and non-streaming generation.
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def load(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=MODEL_SERVER_URL,
            timeout=httpx.Timeout(
                connect=10.0,      # fail fast if server isn't running
                read=REQUEST_TIMEOUT,   # long timeout for generation
                write=30.0,
                pool=10.0,),
        )
        # Verify the model server is reachable
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            logger.info("model_server_connected", extra={"url": MODEL_SERVER_URL})
        except Exception as exc:
            logger.warning("model_server_not_ready", extra={"error": str(exc)})

    async def unload(self) -> None:
        if self._client:
            await self._client.aclose()

    async def stream(
        self,
        messages: list[dict],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> AsyncIterator[str]:
        """
        Yield text chunks from the model server via SSE streaming.

        Parameters
        ----------
        messages : list of {"role": str, "content": str} — full conversation history
        """
        assert self._client is not None, "ModelClient not loaded — call load() first"

        payload = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with self._client.stream("POST", "/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                        chunk = event.get("text", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as exc:
            logger.error(
                "model_stream_error",
                extra={"error": str(exc), "url": f"{MODEL_SERVER_URL}/chat"},
            )
            yield f"[Model server unreachable at {MODEL_SERVER_URL} — {exc}]"
        except httpx.HTTPStatusError as exc:
            logger.error(
                "model_stream_http_error",
                extra={"status": exc.response.status_code, "body": exc.response.text[:200]},
            )
            yield f"[Model server error {exc.response.status_code} — {exc.response.text[:100]}]"

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Non-streaming single-turn generation. Used by smoke tests."""
        assert self._client is not None
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = await self._client.post("/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")
