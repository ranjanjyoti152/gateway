import asyncio
import json

from app import anthropic, core, models, responses


class FakeStream:
    def __init__(self, chunks):
        self.lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aclose(self):
        pass


def _events(gen):
    async def collect():
        return [x async for x in gen]
    out = []
    for raw in asyncio.run(collect()):
        lines = raw.decode().strip().splitlines()
        out.append(json.loads(lines[1][6:]))
    return out


def test_resolve_names():
    models._state["models"] = ["openai/gpt-5.5", "google/gemini-3.5-flash"]
    assert models.resolve("gemini-3.5-flash") == "google/gemini-3.5-flash"
    assert models.resolve("claude-sonnet-4-5") == core.DEFAULT_MODEL
    assert models.resolve("claude-haiku-4-5") == core.SMALL_MODEL
    assert models.resolve("xai/grok-4.7") == "xai/grok-4.7"


def test_anthropic_request_with_tool_history():
    req = anthropic.to_chat_request({
        "model": "openai/gpt-5.5", "max_tokens": 100, "system": [{"type": "text", "text": "sys"}],
        "tools": [{"name": "t", "input_schema": {"$schema": "x", "type": "object"}},
                  {"type": "web_search_20250305", "name": "web_search"}],
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                              {"type": "tool_use", "id": "c1", "name": "t", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "res", "is_error": True},
                                         {"type": "text", "text": "next"}]},
        ]})
    assert req["max_completion_tokens"] == 100 and "max_tokens" not in req
    assert [t["function"]["name"] for t in req["tools"]] == ["t"]
    assert "$schema" not in req["tools"][0]["function"]["parameters"]
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert req["messages"][3]["content"].startswith("[tool error]")


def test_anthropic_stream_tool_call_with_gemini_stop():
    chunks = [
        {"choices": [{"delta": {"content": "Let me"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]
    ev = _events(anthropic.stream_messages(FakeStream(chunks), "m"))
    types = [e["type"] for e in ev]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    starts = [e["content_block"]["type"] for e in ev if e["type"] == "content_block_start"]
    assert starts == ["text", "tool_use"]
    delta = next(e for e in ev if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["output_tokens"] == 5


def test_responses_request_merges_parallel_calls():
    req = responses.to_chat_request({
        "model": "deepseek-ai/deepseek-v4-pro", "instructions": "be brief",
        "tools": [{"type": "function", "name": "sh", "parameters": {"type": "object"}}, {"type": "web_search"}],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "a", "name": "sh", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "sh", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "x"},
            {"type": "function_call_output", "call_id": "b", "output": "y"},
        ]})
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "system", "user", "assistant", "tool", "tool"]
    assert len(req["messages"][3]["tool_calls"]) == 2
    assert len(req["tools"]) == 1


def test_responses_stream_function_call():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "sh", "arguments": "{\"cmd\""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":\"ls\"}"}}]}, "finish_reason": "tool_calls"}]},
    ]
    ev = _events(responses.stream_response(FakeStream(chunks), "m"))
    done = ev[-1]
    assert done["type"] == "response.completed"
    item = done["response"]["output"][0]
    assert item["type"] == "function_call" and item["arguments"] == '{"cmd":"ls"}' and item["call_id"] == "c1"
    assert [e["sequence_number"] for e in ev] == list(range(1, len(ev) + 1))


def test_openai_max_tokens_rewrite():
    assert core.prepare_chat({"model": "openai/gpt-5", "max_tokens": 5}) == {"model": "openai/gpt-5", "max_completion_tokens": 5}
    assert core.prepare_chat({"model": "xai/grok-4.7", "max_tokens": 5})["max_tokens"] == 5


def test_scrape_filter():
    assert not models._is_model_id("livekit/agents-plugin-openai")
    assert not models._is_model_id("google/index.html")
    assert not models._is_model_id("xai/responses")
    assert models._is_model_id("openai/gpt-5.5")


def test_xai_content_parts():
    body = core.prepare_chat({"model": "xai/grok-4.7", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
        {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "out"}]}]})
    assert body["messages"][0]["content"] == [{"type": "input_text", "text": "hi"},
                                              {"type": "input_image", "image_url": "data:image/png;base64,AA"}]
    assert body["messages"][1]["content"] == "out"


def test_pricing_parse_and_cost(tmp_path, monkeypatch):
    from app import usage
    entry = {"model_id": "openai/gpt-5.5", "provider_id": "openai", "pricing_current": {"rates": [
        {"metric": m, "unit": {"per": 1000000}, "unit_price": {"ship": {"amount": a}}}
        for m, a in [("input_tokens", "10.00"), ("cached_input_tokens", "1.00"), ("output_tokens", "45.00")]]}}
    payload = json.dumps({"llmModels": [entry]}, separators=(",", ":"))
    html = 'self.__next_f.push([1,"3b:' + payload.replace('"', '\\"') + '"])'
    prices = usage.parse_pricing(html)
    assert prices["openai/gpt-5.5"]["openai"] == {"input_tokens": 10.0, "cached_input_tokens": 1.0, "output_tokens": 45.0}

    monkeypatch.setattr(core, "DATA_DIR", str(tmp_path))
    usage._prices = prices
    usage._state.update(usd=0.0, requests=0, by_model={}, by_day={}, input_tokens=0, cached_tokens=0, output_tokens=0)
    usage.record("openai/gpt-5.5", "openai/standard_openai",
                 {"prompt_tokens": 1_000_000, "prompt_tokens_details": {"cached_tokens": 200_000}, "completion_tokens": 100_000})
    # 800k*10 + 200k*1 + 100k*45 per 1M = 8 + 0.2 + 4.5
    assert abs(usage._state["usd"] - 12.7) < 1e-9
    usage.record("openai/gpt-5.5", "", {"input_tokens": 1000, "output_tokens": 0})   # Responses API shape
    s = usage.summary()
    assert s["requests"] == 2 and abs(s["remaining_usd"] - (usage.CREDIT_TOTAL_USD - 12.71)) < 1e-6
    usage.flush()
    assert (tmp_path / "usage.json").exists()


def test_stream_usage_extraction():
    assert core.usage_from_event('{"choices":[],"usage":{"prompt_tokens":3}}') == {"prompt_tokens": 3}
    assert core.usage_from_event('{"type":"response.completed","response":{"usage":{"input_tokens":5}}}') == {"input_tokens": 5}
    assert core.usage_from_event('{"choices":[{"delta":{"content":"hi"}}]}') is None
    assert core.ensure_stream_usage({"stream": True})["stream_options"] == {"include_usage": True}


def test_fix_rejected_reasoning_effort():
    body = {"model": "xai/grok-4.7", "reasoning_effort": "none", "messages": []}
    fixed = core._fix_rejected(body, 400, '{"error":{"message":"This model does not support `reasoning_effort` value `none`."}}')
    assert fixed["reasoning_effort"] == "minimal"
    fixed = core._fix_rejected({**body, "reasoning_effort": "minimal"}, 400,
                               "Model grok-4.20 does not support parameter reasoningEffort")
    assert "reasoning_effort" not in fixed
    resp_body = {"model": "xai/grok-4.7", "reasoning": {"effort": "none", "summary": "auto"}}
    fixed = core._fix_rejected(resp_body, 400, "does not support `reasoning_effort` value `none`")
    assert fixed["reasoning"] == {"effort": "minimal", "summary": "auto"}
    assert core._fix_rejected({"model": "x", "max_tokens": 64000}, 400, "max_tokens is too large")["model"] == "x"
    assert core._fix_rejected({"model": "x"}, 400, "something else") is None
    assert core._fix_rejected({"model": "x", "reasoning_effort": "low"}, 500, "reasoning_effort") is None
