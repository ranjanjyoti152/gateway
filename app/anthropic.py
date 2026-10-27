"""Anthropic Messages API (/v1/messages) on top of OpenAI Chat Completions, for Claude Code."""
import json
import uuid

from . import core, models

_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
         "function_call": "tool_use", "content_filter": "refusal"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def _image_part(block: dict) -> dict:
    src = block.get("source", {})
    url = src.get("url") if src.get("type") == "url" else f"data:{src.get('media_type')};base64,{src.get('data')}"
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_content(block: dict) -> str:
    content = block.get("content")
    text = content if isinstance(content, str) else "\n".join(
        b.get("text", "") if b.get("type") == "text" else "[image omitted]" for b in content or [])
    return f"[tool error]\n{text}" if block.get("is_error") else text


def to_chat_messages(body: dict) -> list[dict]:
    out = []
    system = body.get("system")
    if system:
        out.append({"role": "system", "content": _text(system)})
    for msg in body.get("messages", []):
        role, content = msg["role"], msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            m = {"role": "assistant", "content": text or None}
            if calls:
                m["tool_calls"] = calls
            out.append(m)
            continue
        # user turn: tool results become `tool` messages, the rest stays a user message
        parts = []
        for b in content:
            t = b.get("type")
            if t == "tool_result":
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": _tool_result_content(b)})
            elif t == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif t == "image":
                parts.append(_image_part(b))
        if parts:
            only_text = all(p["type"] == "text" for p in parts)
            out.append({"role": "user", "content": "".join(p["text"] for p in parts) if only_text else parts})
    return out


def _clean_schema(schema):
    """Drop JSON-schema keys some upstream providers (Gemini) reject."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k not in ("$schema", "format")}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def to_chat_request(body: dict) -> dict:
    req = {"model": models.resolve(body.get("model")), "messages": to_chat_messages(body)}
    if body.get("max_tokens"):
        req["max_completion_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p"):
        if k in body:
            req[k] = body[k]
    if body.get("stop_sequences"):
        req["stop"] = body["stop_sequences"]
    tools = [t for t in body.get("tools") or [] if "input_schema" in t]  # skip server tools (web_search etc.)
    if tools:
        req["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": _clean_schema(t["input_schema"])}} for t in tools]
        choice = body.get("tool_choice") or {}
        if choice.get("type") == "any":
            req["tool_choice"] = "required"
        elif choice.get("type") == "tool":
            req["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif choice.get("type") == "none":
            req["tool_choice"] = "none"
    return core.prepare_chat(req)


def _usage(u: dict | None) -> dict:
    u = u or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"input_tokens": (u.get("prompt_tokens") or 0) - cached, "output_tokens": u.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


def _parse_args(raw: str) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"_raw": raw}


def from_chat_response(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        content.append({"type": "tool_use", "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name", ""), "input": _parse_args(fn.get("arguments"))})
    stop = "tool_use" if msg.get("tool_calls") else _STOP.get(choice.get("finish_reason"), "end_turn")
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": None, "usage": _usage(resp.get("usage"))}


async def stream_messages(upstream, model: str, billed_model: str = ""):
    """Translate an OpenAI chat-completions SSE stream into Anthropic message events."""
    yield core.sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None, "usage": _usage(None)}})
    index = -1          # current Anthropic content block index
    open_kind = None    # "text" or ("tool", upstream_tool_index)
    finish, usage, saw_tool = None, None, False
    try:
        async for data in core.iter_sse_data(upstream):
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    if open_kind != "text":
                        if open_kind is not None:
                            yield core.sse("content_block_stop", {"type": "content_block_stop", "index": index})
                        index += 1
                        open_kind = "text"
                        yield core.sse("content_block_start", {"type": "content_block_start", "index": index,
                                                               "content_block": {"type": "text", "text": ""}})
                    yield core.sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                           "delta": {"type": "text_delta", "text": delta["content"]}})
                for tc in delta.get("tool_calls") or []:
                    key = ("tool", tc.get("index", 0))
                    fn = tc.get("function") or {}
                    if open_kind != key:
                        if open_kind is not None:
                            yield core.sse("content_block_stop", {"type": "content_block_stop", "index": index})
                        index += 1
                        open_kind = key
                        saw_tool = True
                        yield core.sse("content_block_start", {"type": "content_block_start", "index": index,
                            "content_block": {"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                              "name": fn.get("name", ""), "input": {}}})
                    if fn.get("arguments"):
                        yield core.sse("content_block_delta", {"type": "content_block_delta", "index": index,
                            "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except Exception as e:
        core.log.warning("stream interrupted: %s", e)
        yield core.sse("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}})
        return
    finally:
        await upstream.aclose()
        core.report_usage(billed_model or model, core.deployment_of(upstream), usage)
    if open_kind is not None:
        yield core.sse("content_block_stop", {"type": "content_block_stop", "index": index})
    # Gemini reports finish_reason "stop" even when it called a tool.
    stop = "tool_use" if saw_tool else _STOP.get(finish, "end_turn")
    yield core.sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                     "usage": _usage(usage)})
    yield core.sse("message_stop", {"type": "message_stop"})


def estimate_tokens(body: dict) -> int:
    """Rough count for /v1/messages/count_tokens (no upstream tokenizer available)."""
    size = len(json.dumps(body.get("system") or "")) + len(json.dumps(body.get("messages") or []))
    size += len(json.dumps(body.get("tools") or []))
    return max(1, size // 4)
