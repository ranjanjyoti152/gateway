"""OpenAI- and Anthropic-compatible gateway in front of LiveKit Inference."""
import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from . import anthropic, core, models, responses, usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(_app):
    if not (core.LIVEKIT_API_KEY and core.LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set")
    if not core.GATEWAY_API_KEYS:
        raise RuntimeError("GATEWAY_API_KEYS must be set; this gateway is exposed publicly")
    models.load_initial()
    usage.load()
    tasks = [asyncio.create_task(models.refresh_loop()), asyncio.create_task(usage.loop())]
    yield
    for t in tasks:
        t.cancel()
    usage.flush()
    await core.client.aclose()


app = FastAPI(title="LiveKit LLM Gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

_PUBLIC = {"/health", "/", "/favicon.ico", "/credits"}
_LANDING = (Path(__file__).parent / "static" / "index.html").read_text()


def _is_anthropic(path: str) -> bool:
    return path.startswith("/v1/messages")


def _error(path: str, status: int, message: str) -> JSONResponse:
    if _is_anthropic(path):
        kind = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
                429: "rate_limit_error"}.get(status, "api_error")
        return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)
    return JSONResponse({"error": {"message": message, "type": "gateway_error", "code": status}}, status_code=status)


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path in _PUBLIC:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    key = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not key or not any(hmac.compare_digest(key, k) for k in core.GATEWAY_API_KEYS):
        return _error(request.url.path, 401, "invalid API key")
    start = time.time()
    resp = await call_next(request)
    log.info("%s %s -> %s (%.0fms)", request.method, request.url.path, resp.status_code, (time.time() - start) * 1000)
    return resp


@app.exception_handler(core.UpstreamError)
async def upstream_error(request: Request, exc: core.UpstreamError):
    log.warning("upstream %s on %s (model=%s): %s", exc.status, request.url.path,
                getattr(request.state, "model", "?"), exc.message[:200])
    return _error(request.url.path, exc.status if exc.status < 500 else 502, exc.message)


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise core.UpstreamError(400, json.dumps({"error": {"message": "request body must be a JSON object"}}))
    return body


def _sse(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
async def landing():
    # Only public model metadata goes into the page; keys and secrets never do.
    catalog = {"models": [{**m, "price": usage.price_summary(m["id"])} for m in models.catalog()],
               "updated": models.status()["updated"], "credits": usage.summary()}
    data = json.dumps(catalog).replace("</", "<\\/")
    return HTMLResponse(_LANDING.replace("/*__CATALOG__*/null", data, 1),
                        headers={"Cache-Control": "public, max-age=300"})


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/credits")
async def credits():
    return JSONResponse(usage.summary(), headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return {"status": "ok", "models": models.status()}


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    data = [{"id": m["id"], "object": "model", "type": "model", "created": now, "created_at": "2025-01-01T00:00:00Z",
             "owned_by": m["provider"], "display_name": m["id"], "supports_tools": m["tools"],
             "supports_vision": m["vision"]}
            for m in models.catalog()]
    return {"object": "list", "data": data, "has_more": False,
            "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}


@app.post("/v1/models/refresh")
async def refresh_models():
    asyncio.create_task(models.refresh())
    return {"status": "refresh started", **models.status()}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _json(request)
    request.state.model = body["model"] = models.resolve(body.get("model"))
    body = core.ensure_stream_usage(core.prepare_chat(body))
    if body["model"].startswith("google/") and body.get("tools"):
        body["tools"] = anthropic._clean_schema(body["tools"])
    if body.get("stream"):
        return _sse(core.passthrough(await core.open_stream("/chat/completions", body), body["model"]))
    return await core.post_json("/chat/completions", body)


@app.post("/v1/responses")
async def responses_api(request: Request):
    body = await _json(request)
    request.state.model = model = models.resolve(body.get("model"))
    if model.startswith("openai/"):
        # LiveKit serves the Responses API natively for OpenAI models.
        body["model"] = model
        if body.get("stream"):
            return _sse(core.passthrough(await core.open_stream("/responses", body), model))
        return await core.post_json("/responses", body)
    chat = responses.to_chat_request(body)
    if body.get("stream"):
        chat.update(stream=True, stream_options={"include_usage": True})
        return _sse(responses.stream_response(await core.open_stream("/chat/completions", chat), model, chat["model"]))
    return responses.from_chat_response(await core.post_json("/chat/completions", chat), model)


@app.post("/v1/messages")
async def messages(request: Request):
    body = await _json(request)
    chat = anthropic.to_chat_request(body)
    model = body.get("model") or chat["model"]
    request.state.model = f"{body.get('model')} -> {chat['model']}"
    if body.get("stream"):
        chat.update(stream=True, stream_options={"include_usage": True})
        return _sse(anthropic.stream_messages(await core.open_stream("/chat/completions", chat), model, chat["model"]))
    return anthropic.from_chat_response(await core.post_json("/chat/completions", chat), model)


@app.get("/v1/embeddings/models")
async def embedding_models():
    # Some VS Code extensions (Kilo, Roo) look for embedding models; LiveKit Inference has none.
    return {"object": "list", "data": []}


@app.post("/v1/embeddings")
async def embeddings():
    return _error("/v1/embeddings", 404, "Embeddings are not available: LiveKit Inference serves chat models only. "
                  "Turn off codebase indexing in your editor extension, or point it at a separate embeddings provider.")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    return {"input_tokens": anthropic.estimate_tokens(await _json(request))}
