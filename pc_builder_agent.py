"""
pc_builder_agent.py — same loop, new tool: it specs a PC for your use case
and searches second-hand marketplaces for parts and rough prices.

    pip install anthropic ddgs          # (older name: duckduckgo_search)
    export ANTHROPIC_API_KEY=sk-ant-...
    python pc_builder_agent.py

Note: web search gives links + approximate prices, not a clean price feed.
Always open the listing to confirm price, condition, and shipping.
"""

import json
import anthropic

# Works with either the new 'ddgs' package or the older 'duckduckgo_search'.
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You are a PC-building advisor. For a use case and budget, pick a sensible "
    "parts list (CPU, GPU, RAM, storage, motherboard, PSU, case, cooler), check "
    "basic compatibility (CPU socket, PSU wattage), and use search_listings to "
    "find second-hand prices on marketplaces like eBay, Craigslist, Facebook "
    "Marketplace, and r/hardwareswap. Report rough prices with links, and remind "
    "the user to open each listing to verify price, condition, and shipping."
)


def search_listings(query: str, max_results: int = 8) -> dict:
    with DDGS() as ddgs:
        hits = ddgs.text(query, max_results=max_results)
    return {
        "query": query,
        "results": [
            {"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")}
            for h in hits
        ],
    }


TOOLS = [{
    "name": "search_listings",
    "description": "Search the web (including marketplaces) for parts and prices. "
                   "Include terms like 'used', 'ebay', or a part name in the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "e.g. 'used RTX 4070 ebay' or 'second hand Ryzen 5 7600'."}
        },
        "required": ["query"],
    },
}]


def run_agent(messages: list) -> str:
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=2048, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"  [searching] {b.input.get('query')}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(search_listings(**b.input)),
                })
        messages.append({"role": "user", "content": results})


def main():
    print("PC-builder agent. Tell it your use case and budget. Type 'quit' to exit.\n")
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