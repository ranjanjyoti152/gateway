"""Config, LiveKit inference token minting and upstream HTTP helpers."""
import json
import logging
import os
import time

import httpx
import jwt

log = logging.getLogger("gateway")

UPSTREAM_URL = os.environ.get("LIVEKIT_INFERENCE_URL", "https://agent-gateway.livekit.cloud/v1").rstrip("/")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
GATEWAY_API_KEYS = {k.strip() for k in os.environ.get("GATEWAY_API_KEYS", "").split(",") if k.strip()}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openai/gpt-5.5")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "openai/gpt-5.4-mini")
MODEL_ALIASES: dict = json.loads(os.environ.get("MODEL_ALIASES") or "{}")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_REFRESH_HOURS = float(os.environ.get("MODEL_REFRESH_HOURS", "6"))
PROBE_MODELS = os.environ.get("PROBE_MODELS", "true").lower() != "false"
EXTRA_MODELS = [m.strip() for m in os.environ.get("EXTRA_MODELS", "").split(",") if m.strip()]

_TOKEN_TTL = 3600
_token: tuple[str, float] = ("", 0.0)


def livekit_token() -> str:
    """Short-lived JWT with the inference grant, re-minted 5 minutes before expiry."""
    global _token
    tok, exp = _token
    if time.time() > exp - 300:
        now = int(time.time())
        claims = {"iss": LIVEKIT_API_KEY, "sub": "", "nbf": now, "exp": now + _TOKEN_TTL,
                  "inference": {"perform": True}}
        tok = jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")
        _token = (tok, now + _TOKEN_TTL)
    return tok


client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))

# Set by app.usage; called as usage_hook(model, deployment, usage_dict) for every billed upstream call.
usage_hook = None


def report_usage(model: str, deployment: str, usage: dict | None) -> None:
    if usage_hook and usage:
        try:
            usage_hook(model, deployment or "", usage)
        except Exception:
            log.exception("usage recording failed")


def deployment_of(r) -> str:
    """Provider that served the call, e.g. "openai/standard_openai"; picks the right price when several exist."""
    headers = getattr(r, "headers", None) or {}
    return headers.get("x-livekit-inference-llm-deployment", "")


class UpstreamError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(body)
        self.status = status
        self.body = body

    @property
    def message(self) -> str:
        try:
            err = json.loads(self.body).get("error")
            return err.get("message", str(err)) if isinstance(err, dict) else str(err)
        except Exception:
            return self.body.strip() or f"upstream error {self.status}"


def _headers() -> dict:
    return {"Authorization": f"Bearer {livekit_token()}", "Content-Type": "application/json"}


_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _without_token_limit(body: dict, status: int, text: str) -> dict | None:
    """Clients like Claude Code ask for 32k+ output tokens; drop the limit if a model rejects it."""
    if status != 400 or "token" not in text.lower() or not any(k in body for k in _LIMIT_KEYS):
        return None
    log.info("upstream rejected token limit for %s, retrying without it", body.get("model"))
    return {k: v for k, v in body.items() if k not in _LIMIT_KEYS}


async def post_json(path: str, body: dict, track: bool = True) -> dict:
    for attempt in range(2):
        r = await client.post(UPSTREAM_URL + path, json=body, headers=_headers())
        if r.status_code < 400:
            data = r.json()
            if track:
                report_usage(body.get("model", ""), deployment_of(r), data.get("usage"))
            return data
        retry = _without_token_limit(body, r.status_code, r.text) if attempt == 0 else None
        if retry is None:
            raise UpstreamError(r.status_code, r.text)
        body = retry
    raise AssertionError("unreachable")


async def open_stream(path: str, body: dict) -> httpx.Response:
    """Open a streaming request; raises UpstreamError before any bytes reach the client."""
    for attempt in range(2):
        req = client.build_request("POST", UPSTREAM_URL + path, json=body, headers=_headers())
        r = await client.send(req, stream=True)
        if r.status_code < 400:
            return r
        text = (await r.aread()).decode(errors="replace")
        await r.aclose()
        retry = _without_token_limit(body, r.status_code, text) if attempt == 0 else None
        if retry is None:
            raise UpstreamError(r.status_code, text)
        body = retry
    raise AssertionError("unreachable")


async def iter_sse_data(resp: httpx.Response):
    """Yield the payload of each `data:` line from an upstream SSE stream."""
    async for line in resp.aiter_lines():
        if line.startswith("data:"):
            yield line[5:].strip()


def usage_from_event(data: str) -> dict | None:
    """Usage block from a chat chunk or a Responses `response.completed` event, if present."""
    if '"usage"' not in data:
        return None
    try:
        ev = json.loads(data)
    except ValueError:
        return None
    if isinstance(ev.get("usage"), dict):
        return ev["usage"]
    resp = ev.get("response")
    if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
        return resp["usage"]
    return None


async def passthrough(resp: httpx.Response, model: str):
    """Forward upstream SSE bytes unchanged while picking out the final usage for billing."""
    deployment, usage, buf = deployment_of(resp), None, b""
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.startswith(b"data:"):
                    usage = usage_from_event(line[5:].strip().decode(errors="replace")) or usage
    finally:
        await resp.aclose()
        report_usage(model, deployment, usage)


def ensure_stream_usage(body: dict) -> dict:
    """Ask for usage in the final chunk so streamed calls can be priced."""
    if body.get("stream") and "stream_options" not in body:
        body = {**body, "stream_options": {"include_usage": True}}
    elif body.get("stream"):
        body = {**body, "stream_options": {**body["stream_options"], "include_usage": True}}
    return body


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _xai_part(part: dict) -> dict:
    if part.get("type") == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if part.get("type") == "image_url":
        img = part.get("image_url")
        return {"type": "input_image", "image_url": img.get("url") if isinstance(img, dict) else img}
    return part


def _xai_messages(messages: list) -> list:
    """LiveKit routes xAI through the Responses API, which rejects chat-style content parts."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            if m.get("role") in ("tool", "assistant", "system"):
                text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                m = {**m, "content": text}
            else:
                m = {**m, "content": [_xai_part(p) for p in content]}
        out.append(m)
    return out


def prepare_chat(body: dict) -> dict:
    """Upstream quirks: OpenAI reasoning models reject `max_tokens`; xAI needs Responses-style content parts."""
    model = body.get("model", "")
    if model.startswith("openai/") and "max_tokens" in body:
        body = dict(body)
        limit = body.pop("max_tokens")
        body.setdefault("max_completion_tokens", limit)
    if model.startswith("xai/") and any(isinstance(m.get("content"), list) for m in body.get("messages") or []):
        body = {**body, "messages": _xai_messages(body["messages"])}
    return body
