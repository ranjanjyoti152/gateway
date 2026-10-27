"""OpenAI Responses API (/v1/responses) on top of Chat Completions, for models LiveKit only serves via chat (Codex)."""
import json
import time
import uuid

from . import core, models
from .anthropic import _clean_schema


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if p.get("type") in ("input_text", "output_text", "text"))


def _user_parts(content):
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if p.get("type") in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "input_image" and p.get("image_url"):
            parts.append({"type": "image_url", "image_url": {"url": p["image_url"]}})
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def to_chat_request(body: dict) -> dict:
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for it in items or []:
        t = it.get("type", "message")
        if t == "message":
            role = {"developer": "system"}.get(it.get("role"), it.get("role", "user"))
            if role == "assistant":
                msgs.append({"role": "assistant", "content": _content_text(it.get("content"))})
            elif role == "system":
                msgs.append({"role": "system", "content": _content_text(it.get("content"))})
            else:
                msgs.append({"role": role, "content": _user_parts(it.get("content"))})
        elif t == "function_call":
            call = {"id": it.get("call_id"), "type": "function",
                    "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}}
            # merge consecutive calls into one assistant turn (parallel tool calls)
            if msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("tool_calls") is not None:
                msgs[-1]["tool_calls"].append(call)
            else:
                msgs.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif t == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": out if isinstance(out, str) else _content_text(out) or json.dumps(out)})
        # reasoning items and others carry no state chat models can use
    req = {"model": models.resolve(body.get("model")), "messages": msgs}
    if body.get("max_output_tokens"):
        req["max_completion_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "parallel_tool_calls"):
        if k in body:
            req[k] = body[k]
    effort = (body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else None
    if effort:
        req["reasoning_effort"] = effort
    tools = [t for t in body.get("tools") or [] if t.get("type") == "function"]
    if tools:
        req["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                         "parameters": _clean_schema(t.get("parameters") or {"type": "object", "properties": {}})}}
                        for t in tools]
        choice = body.get("tool_choice")
        if choice in ("auto", "required", "none"):
            req["tool_choice"] = choice
        elif isinstance(choice, dict) and choice.get("name"):
            req["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    else:
        req.pop("parallel_tool_calls", None)
    return core.prepare_chat(req)


def _usage(u: dict | None) -> dict:
    u = u or {}
    return {"input_tokens": u.get("prompt_tokens") or 0, "output_tokens": u.get("completion_tokens") or 0,
            "total_tokens": u.get("total_tokens") or 0,
            "input_tokens_details": {"cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0},
            "output_tokens_details": {"reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0}}


def _response(rid: str, model: str, output: list, status: str, usage=None) -> dict:
    return {"id": rid, "object": "response", "created_at": int(time.time()), "status": status, "model": model,
            "output": output, "usage": _usage(usage) if usage is not None else None, "error": None}


def _msg_item(text: str, status="completed") -> dict:
    return {"type": "message", "id": f"msg_{uuid.uuid4().hex}", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _call_item(call_id: str, name: str, args: str, status="completed") -> dict:
    return {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}", "call_id": call_id, "name": name,
            "arguments": args, "status": status}


def from_chat_response(resp: dict, model: str) -> dict:
    msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
    output = []
    if msg.get("content"):
        output.append(_msg_item(msg["content"]))
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        output.append(_call_item(c.get("id") or f"call_{uuid.uuid4().hex[:24]}", fn.get("name", ""), fn.get("arguments") or "{}"))
    return _response(f"resp_{uuid.uuid4().hex}", model, output, "completed", resp.get("usage") or {})


async def stream_response(upstream, model: str, billed_model: str = ""):
    rid = f"resp_{uuid.uuid4().hex}"
    seq = 0

    def ev(kind: str, **data):
        nonlocal seq
        seq += 1
        return core.sse(kind, {"type": kind, "sequence_number": seq, **data})

    yield ev("response.created", response=_response(rid, model, [], "in_progress"))
    yield ev("response.in_progress", response=_response(rid, model, [], "in_progress"))
    output: list[dict] = []
    text_item = None            # (output_index, item)
    calls: dict[int, tuple[int, dict]] = {}   # upstream tool index -> (output_index, item)
    usage = None

    def close_text():
        nonlocal text_item
        if text_item is None:
            return []
        oi, item = text_item
        text_item = None
        text = item["content"][0]["text"]
        item["status"] = "completed"
        return [ev("response.output_text.done", item_id=item["id"], output_index=oi, content_index=0, text=text),
                ev("response.content_part.done", item_id=item["id"], output_index=oi, content_index=0,
                   part=item["content"][0]),
                ev("response.output_item.done", output_index=oi, item=item)]

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
                    if text_item is None:
                        item = _msg_item("", "in_progress")
                        output.append(item)
                        text_item = (len(output) - 1, item)
                        yield ev("response.output_item.added", output_index=text_item[0],
                                 item={**item, "content": []})
                        yield ev("response.content_part.added", item_id=item["id"], output_index=text_item[0],
                                 content_index=0, part={"type": "output_text", "text": "", "annotations": []})
                    oi, item = text_item
                    item["content"][0]["text"] += delta["content"]
                    yield ev("response.output_text.delta", item_id=item["id"], output_index=oi, content_index=0,
                             delta=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    if idx not in calls:
                        for e in close_text():
                            yield e
                        item = _call_item(tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", fn.get("name", ""), "",
                                          "in_progress")
                        output.append(item)
                        calls[idx] = (len(output) - 1, item)
                        yield ev("response.output_item.added", output_index=calls[idx][0], item=item)
                    oi, item = calls[idx]
                    if fn.get("arguments"):
                        item["arguments"] += fn["arguments"]
                        yield ev("response.function_call_arguments.delta", item_id=item["id"], output_index=oi,
                                 delta=fn["arguments"])
    except Exception as e:
        core.log.warning("stream interrupted: %s", e)
        yield ev("response.failed", response={**_response(rid, model, output, "failed"),
                                              "error": {"code": "server_error", "message": str(e)}})
        return
    finally:
        await upstream.aclose()
        core.report_usage(billed_model or model, core.deployment_of(upstream), usage)
    for e in close_text():
        yield e
    for oi, item in calls.values():
        item["status"] = "completed"
        item["arguments"] = item["arguments"] or "{}"
        yield ev("response.function_call_arguments.done", item_id=item["id"], output_index=oi,
                 arguments=item["arguments"])
        yield ev("response.output_item.done", output_index=oi, item=item)
    yield ev("response.completed", response=_response(rid, model, output, "completed", usage or {}))
