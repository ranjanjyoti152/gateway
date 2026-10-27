"""Model discovery: scrape LiveKit docs for model IDs, probe each against the gateway, cache the result."""
import asyncio
import base64
import json
import logging
import os
import re
import struct
import time
import zlib

from . import core

log = logging.getLogger("gateway.models")

DOCS_BASE = "https://docs.livekit.io/agents/models/llm/"
# Provider prefixes LiveKit Inference uses for model IDs.
_PROVIDERS = r"(?:openai|google|xai|deepseek-ai|moonshotai|qwen|meta-llama|meta|anthropic|mistralai|zai-org|minimax|cerebras|livekit)"
_MODEL_RE = re.compile(rf"\b{_PROVIDERS}/[a-z0-9][a-z0-9.\-]*[a-z0-9]\b")
_SUBPAGE_RE = re.compile(r'href="/agents/models/llm/([a-z0-9\-]+)/')
_NOT_MODELS = {"livekit/agents", "livekit/livekit", "livekit/plugins", "livekit/agents-js"}

# Last-known-good list so the gateway works even if the docs layout changes.
_FALLBACK = [
    "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5", "openai/gpt-5-mini",
    "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o", "openai/gpt-4o-mini",
    "google/gemini-2.5-flash", "google/gemini-3.5-flash", "deepseek-ai/deepseek-v3",
]

_state = {"models": [], "info": {}, "updated": 0.0, "source": "none"}
_lock = asyncio.Lock()


def _cache_path() -> str:
    return os.path.join(core.DATA_DIR, "models.json")


def _is_model_id(mid: str) -> bool:
    return (mid not in _NOT_MODELS and not mid.startswith("livekit/agents-plugin")
            and not mid.endswith((".html", "/index")) and not mid.split("/", 1)[1] in {"index", "responses", "plugins"})


async def scrape_docs() -> list[str]:
    r = await core.client.get(DOCS_BASE, follow_redirects=True, timeout=30)
    r.raise_for_status()
    pages = [r.text]
    subpages = sorted(set(_SUBPAGE_RE.findall(r.text)))
    results = await asyncio.gather(
        *(core.client.get(DOCS_BASE + p + "/", follow_redirects=True, timeout=30) for p in subpages),
        return_exceptions=True)
    pages += [x.text for x in results if not isinstance(x, Exception) and x.status_code == 200]
    found = {m for page in pages for m in _MODEL_RE.findall(page)}
    return sorted(m for m in found if _is_model_id(m))


_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def _solid_png(rgb: tuple[int, int, int], size: int = 256) -> str:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


# Text-only models guess a color, so require two different colors to both be named correctly.
_VISION_TESTS = [(_solid_png((220, 20, 20)), "red"), (_solid_png((20, 40, 220)), "blue")]


async def _sees(model: str) -> bool:
    # One retry: a capable model occasionally misreads a single sample.
    return await _sees_once(model) or await _sees_once(model)


async def _sees_once(model: str) -> bool:
    for url, color in _VISION_TESTS:
        resp = await _call({"model": model, "max_completion_tokens": 300, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What single color fills this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": url}}]}]})
        text = (((resp or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if color not in text.lower():
            return False
    return True


async def _call(body: dict) -> dict | None:
    """One upstream call with a retry for cold starts; None means the model is not usable."""
    for _ in range(2):
        try:
            return await asyncio.wait_for(core.post_json("/chat/completions", core.prepare_chat(body)), timeout=90)
        except core.UpstreamError as e:
            if e.status in (400, 404):
                return None
        except Exception:
            pass
    return None


async def _probe(model: str, sem: asyncio.Semaphore) -> dict | None:
    """Returns {"tools": bool, "vision": bool} for a working model, None otherwise."""
    async with sem:
        if await _call({"model": model, "messages": [{"role": "user", "content": "hi"}],
                        "max_completion_tokens": 300}) is None:
            return None
        resp = await _call({"model": model, "tools": _TOOL, "max_completion_tokens": 500,
                            "messages": [{"role": "user", "content": "What is the weather in Delhi? Use the tool."}]})
        vision = await _sees(model)
    msg = ((resp or {}).get("choices") or [{}])[0].get("message") or {}
    return {"tools": bool(msg.get("tool_calls")), "vision": vision}


async def refresh() -> None:
    async with _lock:
        try:
            candidates = await scrape_docs()
            source = "docs"
        except Exception as e:
            log.warning("docs scrape failed (%s); using cached/fallback list", e)
            candidates, source = (_state["models"] or _load_cache() or _FALLBACK), "cache"
        candidates = sorted(set(candidates) | set(core.EXTRA_MODELS))
        if not candidates:
            candidates = _FALLBACK
        if core.PROBE_MODELS:
            sem = asyncio.Semaphore(12)
            results = await asyncio.gather(*(_probe(m, sem) for m in candidates))
            info = {m: r for m, r in zip(candidates, results) if r is not None}
        else:
            info = {m: _state["info"].get(m, {"tools": True, "vision": False}) for m in candidates}
        working = sorted(info)
        if not working:
            log.warning("no models passed probing; keeping previous list")
            return
        _state.update(models=working, info=info, updated=time.time(), source=source)
        log.info("model list refreshed: %d working (%d tools, %d vision) of %d candidates", len(working),
                 sum(i["tools"] for i in info.values()), sum(i["vision"] for i in info.values()), len(candidates))
        try:
            os.makedirs(core.DATA_DIR, exist_ok=True)
            with open(_cache_path(), "w") as f:
                json.dump(_state, f, indent=1)
        except OSError as e:
            log.warning("could not write model cache: %s", e)


def _load_cache() -> list[str]:
    try:
        with open(_cache_path()) as f:
            data = json.load(f)
        _state.update(data)
        return data.get("models", [])
    except (OSError, ValueError):
        return []


def load_initial() -> None:
    """Populate the list from cache (or fallback) so requests work before the first refresh ends."""
    if not _load_cache():
        _state.update(models=list(_FALLBACK), source="fallback")


async def refresh_loop() -> None:
    while True:
        try:
            await refresh()
        except Exception:
            log.exception("model refresh failed")
        await asyncio.sleep(core.MODEL_REFRESH_HOURS * 3600)


def available() -> list[str]:
    return list(_state["models"])


def catalog() -> list[dict]:
    """Working models with capability flags, for /v1/models and the landing page."""
    info = _state.get("info") or {}
    return [{"id": m, "provider": m.split("/", 1)[0], "tools": info.get(m, {}).get("tools", True),
             "vision": info.get(m, {}).get("vision", False)} for m in _state["models"]]


def status() -> dict:
    return {"count": len(_state["models"]), "updated": _state["updated"], "source": _state["source"]}


def resolve(name: str | None) -> str:
    """Map whatever a client sends (bare names, Claude names, aliases) to a LiveKit model ID."""
    if not name:
        return core.DEFAULT_MODEL
    if name in core.MODEL_ALIASES:
        return core.MODEL_ALIASES[name]
    models = _state["models"]
    if name in models or "/" in name:
        return name
    for m in models:
        if m.split("/", 1)[1] == name:
            return m
    if name.startswith("claude"):
        # Claude Code asks for Claude model IDs; route them to configured defaults.
        return core.SMALL_MODEL if "haiku" in name else core.DEFAULT_MODEL
    return name
