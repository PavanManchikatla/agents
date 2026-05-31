"""
weather_agent.py — a minimal AI agent, built from scratch (no framework).

This is the agent loop from the architecture diagram, made concrete:

    WORKING MEMORY   ->  the `messages` list
    LLM CORE         ->  client.messages.create(...)        (reason / decide)
    TOOLS            ->  get_weather(...)                   (act on the world)
    GUARDRAILS       ->  input checks + a max-step limit
    OBSERVATION      ->  the tool_result fed back in
    LOOP             ->  the `while True:` in run_agent()

Run it:
    pip install anthropic requests
    export ANTHROPIC_API_KEY=sk-ant-...      # your key
    python weather_agent.py

Then ask things like:
    "What's the weather in Onalaska, Wisconsin?"
    "Is it colder in Oslo or Tokyo right now?"
"""

import os
import json
import requests
import anthropic

# The LLM core. Swap to "claude-opus-4-8" for harder reasoning; sonnet is a
# fast, cheap default that's plenty for tool use like this.
MODEL = "claude-opus-4-8"
MAX_STEPS = 6  # GUARDRAIL: never let the loop run forever.

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ---------------------------------------------------------------------------
# THE TOOL  — the only way this agent can touch the outside world.
# A tool is just a normal function. The model never runs it; *you* do.
# ---------------------------------------------------------------------------

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
    71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def get_weather(location: str) -> dict:
    """Look up current weather for a place name using the free Open-Meteo API."""
    if not location or not location.strip():
        return {"error": "location was empty"}

    # 1) Turn the place name into coordinates (geocoding).
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return {"error": f"could not find a place called {location!r}"}

    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # 2) Fetch the current conditions for those coordinates.
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        },
        timeout=10,
    ).json()
    cur = wx["current"]

    # Return clean, structured data — the model reads this as the "observation".
    return {
        "resolved_location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "temperature_c": cur["temperature_2m"],
        "humidity_percent": cur["relative_humidity_2m"],
        "wind_speed_kmh": cur["wind_speed_10m"],
        "conditions": WEATHER_CODES.get(cur["weather_code"], "unknown"),
    }


# What the model is *told* about the tool. The `input_schema` is how it knows
# what arguments to send. Good descriptions here = good tool calls.
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city or place by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and (optionally) region/country, e.g. 'Oslo' or 'Onalaska, Wisconsin'.",
                }
            },
            "required": ["location"],
        },
    }
]

# Map tool names to the real functions, so the loop can dispatch by name.
TOOL_FUNCTIONS = {"get_weather": get_weather}


# ---------------------------------------------------------------------------
# THE AGENT LOOP
# ---------------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    # WORKING MEMORY: starts with just the user's goal. Grows as we go.
    messages = [{"role": "user", "content": user_message}]

    for step in range(MAX_STEPS):  # GUARDRAIL: capped number of turns
        # LLM CORE: look at everything so far, decide the next move.
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # If the model didn't ask for a tool, it's done — return its answer.
        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        # Otherwise it wants to ACT. Record its turn in working memory.
        messages.append({"role": "assistant", "content": response.content})

        # Run every tool the model asked for and collect the OBSERVATIONS.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool call] {block.name}({json.dumps(block.input)})")
            fn = TOOL_FUNCTIONS[block.name]          # dispatch by name
            result = fn(**block.input)               # actually do the thing
            print(f"  [observation] {json.dumps(result)}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,             # ties result to the request
                "content": json.dumps(result),
            })

        # Feed observations back in — this closes the loop.
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit the maximum number of steps without finishing."


# ---------------------------------------------------------------------------
# A tiny CLI so you can talk to it.
# ---------------------------------------------------------------------------

def main():
    print("Weather agent. Ask about the weather, or type 'quit'.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer = run_agent(question)
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    main()