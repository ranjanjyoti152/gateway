"""Credit tracking: price every upstream call from LiveKit's published rates and keep a running total."""
import asyncio
import json
import logging
import os
import time

from . import core

log = logging.getLogger("gateway.usage")

PRICING_URL = "https://livekit.io/pricing/inference"
CREDIT_TOTAL_USD = float(os.environ.get("CREDIT_TOTAL_USD", "17000"))
# Spend from before this gateway started tracking (copy it from the LiveKit Cloud billing page).
CREDIT_USED_BEFORE_USD = float(os.environ.get("CREDIT_USED_BEFORE_USD", "0"))

_prices: dict[str, dict[str, dict[str, float]]] = {}   # model -> provider -> USD per 1M tokens
_state = {"since": 0.0, "usd": 0.0, "requests": 0, "unpriced": 0,
          "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "by_model": {}, "by_day": {}}
_dirty = False


def _path(name: str) -> str:
    return os.path.join(core.DATA_DIR, name)


def _save(name: str, data) -> None:
    try:
        os.makedirs(core.DATA_DIR, exist_ok=True)
        tmp = _path(name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _path(name))
    except OSError as e:
        log.warning("could not write %s: %s", name, e)


def _load(name: str):
    try:
        with open(_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---- pricing -----------------------------------------------------------------

def parse_pricing(html: str) -> dict:
    """Pull the `llmModels` array out of the pricing page's embedded Next.js payload."""
    i = html.index('\\"llmModels\\":[')
    # The payload is a JS string literal; unescape it so the JSON inside parses.
    chunk = html[i:].encode("latin-1", "backslashreplace").decode("unicode_escape")
    start = chunk.index('"llmModels":') + len('"llmModels":')
    entries, _ = json.JSONDecoder().raw_decode(chunk, start)
    prices: dict = {}
    for e in entries:
        rates = {}
        for r in (e.get("pricing_current") or {}).get("rates", []):
            tier = r["unit_price"].get("ship") or next(iter(r["unit_price"].values()))
            per = (r.get("unit") or {}).get("per") or 1_000_000
            rates[r["metric"]] = float(tier["amount"]) * 1_000_000 / per
        if "input_tokens" in rates and "output_tokens" in rates:
            rates.setdefault("cached_input_tokens", rates["input_tokens"])
            prices.setdefault(e["model_id"], {})[e["provider_id"]] = rates
    return prices


async def refresh_prices() -> None:
    global _prices
    try:
        r = await core.client.get(PRICING_URL, follow_redirects=True, timeout=30)
        r.raise_for_status()
        prices = parse_pricing(r.text)
        if not prices:
            raise ValueError("no prices found")
        _prices = prices
        _save("pricing.json", prices)
        log.info("pricing refreshed for %d models", len(prices))
    except Exception as e:
        log.warning("pricing refresh failed (%s); using cached prices", e)
        _prices = _prices or _load("pricing.json") or {}


def _rates(model: str, provider: str = "") -> dict | None:
    p = _prices.get(model)
    if not p:
        return None
    return p.get(provider) or next(iter(p.values()))


def price_summary(model: str) -> dict | None:
    """Per-1M-token prices for the landing page (cheapest provider when several serve a model)."""
    p = _prices.get(model)
    if not p:
        return None
    r = min(p.values(), key=lambda x: x["input_tokens"] + x["output_tokens"])
    return {"input": r["input_tokens"], "cached": r["cached_input_tokens"], "output": r["output_tokens"]}


# ---- recording ---------------------------------------------------------------

def _normalize(u: dict) -> tuple[int, int, int]:
    if "prompt_tokens" in u:   # chat completions
        return (u.get("prompt_tokens") or 0, (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                u.get("completion_tokens") or 0)
    return (u.get("input_tokens") or 0, (u.get("input_tokens_details") or {}).get("cached_tokens") or 0,
            u.get("output_tokens") or 0)


def record(model: str, deployment: str, usage: dict) -> None:
    global _dirty
    inp, cached, out = _normalize(usage)
    cached = min(cached, inp)
    rates = _rates(model, deployment.split("/", 1)[0])
    usd = 0.0
    if rates:
        usd = ((inp - cached) * rates["input_tokens"] + cached * rates["cached_input_tokens"]
               + out * rates["output_tokens"]) / 1_000_000
    else:
        _state["unpriced"] += 1
    s = _state
    s["usd"] += usd
    s["requests"] += 1
    s["input_tokens"] += inp
    s["cached_tokens"] += cached
    s["output_tokens"] += out
    m = s["by_model"].setdefault(model, {"usd": 0.0, "requests": 0, "input_tokens": 0, "output_tokens": 0})
    m["usd"] += usd
    m["requests"] += 1
    m["input_tokens"] += inp
    m["output_tokens"] += out
    d = s["by_day"].setdefault(time.strftime("%Y-%m-%d", time.gmtime()), {"usd": 0.0, "requests": 0})
    d["usd"] += usd
    d["requests"] += 1
    _dirty = True


def summary() -> dict:
    s = _state
    days = sorted(s["by_day"])
    today = time.strftime("%Y-%m-%d", time.gmtime())
    last7 = [time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400 * k)) for k in range(6, -1, -1)]
    used = CREDIT_USED_BEFORE_USD + s["usd"]
    top = sorted(s["by_model"].items(), key=lambda kv: kv[1]["usd"], reverse=True)[:8]
    return {
        "credit_total_usd": CREDIT_TOTAL_USD,
        "used_before_usd": CREDIT_USED_BEFORE_USD,
        "tracked_usd": round(s["usd"], 6),
        "used_usd": round(used, 6),
        "remaining_usd": round(CREDIT_TOTAL_USD - used, 6),
        "today_usd": round(s["by_day"].get(today, {}).get("usd", 0.0), 6),
        "last_7_days": [{"day": d, "usd": round(s["by_day"].get(d, {}).get("usd", 0.0), 6)} for d in last7],
        "requests": s["requests"],
        "input_tokens": s["input_tokens"],
        "cached_tokens": s["cached_tokens"],
        "output_tokens": s["output_tokens"],
        "unpriced_requests": s["unpriced"],
        "top_models": [{"model": k, "usd": round(v["usd"], 6), "requests": v["requests"]} for k, v in top],
        "tracking_since": s["since"],
        "first_day": days[0] if days else today,
        "updated": time.time(),
    }


# ---- lifecycle ---------------------------------------------------------------

def load() -> None:
    global _prices
    saved = _load("usage.json")
    if saved:
        _state.update(saved)
    if not _state["since"]:
        _state["since"] = time.time()
    _prices = _load("pricing.json") or {}
    core.usage_hook = record


def flush() -> None:
    global _dirty
    if _dirty:
        _dirty = False
        # keep 90 days of daily totals
        for d in sorted(_state["by_day"])[:-90]:
            del _state["by_day"][d]
        _save("usage.json", _state)


async def loop() -> None:
    last_prices = 0.0
    while True:
        if time.time() - last_prices > core.MODEL_REFRESH_HOURS * 3600:
            last_prices = time.time()
            await refresh_prices()
        flush()
        await asyncio.sleep(10)
