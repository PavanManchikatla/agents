"""
stock_agent.py — same loop, new tools: it can look up market data.

This is the seed of your trading project's research layer: read-only data,
no orders. (Real money stays behind a deterministic risk layer, later on.)

    pip install anthropic yfinance
    export ANTHROPIC_API_KEY=sk-ant-...
    python stock_agent.py
"""

import json
import yfinance as yf
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def get_stock_price(symbol: str) -> dict:
    try:
        fi = yf.Ticker(symbol).fast_info
        return {
            "symbol": symbol.upper(),
            "last_price": fi.last_price,
            "previous_close": fi.previous_close,
            "day_high": fi.day_high,
            "day_low": fi.day_low,
            "currency": fi.currency,
        }
    except Exception as e:
        return {"error": f"could not get price for {symbol!r}: {e}"}


def get_recent_prices(symbol: str, days: int = 5) -> dict:
    days = max(1, min(int(days), 30))   # GUARDRAIL: clamp to a sane range
    hist = yf.Ticker(symbol).history(period=f"{days}d")
    if hist.empty:
        return {"error": f"no data for {symbol!r}"}
    return {
        "symbol": symbol.upper(),
        "closes": [{"date": str(d.date()), "close": round(c, 2)} for d, c in hist["Close"].items()],
    }


TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the CURRENT price and day's range for a stock, ETF, or crypto ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker, e.g. 'AAPL', 'SPY', 'BTC-USD'."}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_recent_prices",
        "description": "Get recent daily closing prices to see a trend over the last N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol."},
                "days": {"type": "integer", "description": "How many days back, 1 to 30. Defaults to 5."},
            },
            "required": ["symbol"],
        },
    },
]

TOOL_FUNCTIONS = {"get_stock_price": get_stock_price, "get_recent_prices": get_recent_prices}


def run_agent(messages: list) -> str:
    while True:
        resp = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"  [tool call] {b.name}({json.dumps(b.input)})")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(TOOL_FUNCTIONS[b.name](**b.input)),
                })
        messages.append({"role": "user", "content": results})


def main():
    print("Stock agent. Ask about prices and trends. Type 'quit' to exit.\n")
    conversation = []
    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            conversation.append({"role": "user", "content": q})
            print(f"\nagent> {run_agent(conversation)}\n")


if __name__ == "__main__":
    main()