"""
weather_agent_v2.py — the same from-scratch agent, now with two upgrades:

  NEW 1) SHORT-TERM MEMORY
         The `messages` list now lives in main() and persists across
         questions, so the agent remembers the conversation. Ask "weather in
         Oslo?" then "is it colder than Tokyo?" and it knows what "it" means.

  NEW 2) A SECOND TOOL  (get_forecast)
         Now the model must CHOOSE: get_weather for *right now*, or
         get_forecast for *upcoming days*. It picks based on the tool
         descriptions + your wording. Watch the [tool call] lines to see
         which one it reached for.

Run it:
    pip install anthropic requests
    export ANTHROPIC_API_KEY=sk-ant-...
    python weather_agent_v2.py
"""

import os
import json
import requests
import anthropic

MODEL = "claude-sonnet-4-6"
MAX_STEPS = 6

client = anthropic.Anthropic()

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
    71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# ---------------------------------------------------------------------------
# A small shared helper. Both tools need to turn a name into coordinates,
# so we factor it out instead of duplicating it (good habit for real tools).
# ---------------------------------------------------------------------------

def _geocode(location: str):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return None
    return geo["results"][0]


# ---------------------------------------------------------------------------
# TOOL 1 — current conditions (same as before)
# ---------------------------------------------------------------------------

def get_weather(location: str) -> dict:
    if not location or not location.strip():
        return {"error": "location was empty"}
    place = _geocode(location)
    if place is None:
        return {"error": f"could not find a place called {location!r}"}

    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        },
        timeout=10,
    ).json()
    cur = wx["current"]
    return {
        "resolved_location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "temperature_c": cur["temperature_2m"],
        "humidity_percent": cur["relative_humidity_2m"],
        "wind_speed_kmh": cur["wind_speed_10m"],
        "conditions": WEATHER_CODES.get(cur["weather_code"], "unknown"),
    }


# ---------------------------------------------------------------------------
# TOOL 2 — multi-day forecast (NEW)
# ---------------------------------------------------------------------------

def get_forecast(location: str, days: int = 3) -> dict:
    if not location or not location.strip():
        return {"error": "location was empty"}
    days = max(1, min(int(days), 7))  # GUARDRAIL: clamp to a sane 1..7 range
    place = _geocode(location)
    if place is None:
        return {"error": f"could not find a place called {location!r}"}

    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": days,
            "timezone": "auto",
        },
        timeout=10,
    ).json()
    d = wx["daily"]
    forecast = [
        {
            "date": d["time"][i],
            "high_c": d["temperature_2m_max"][i],
            "low_c": d["temperature_2m_min"][i],
            "conditions": WEATHER_CODES.get(d["weather_code"][i], "unknown"),
        }
        for i in range(len(d["time"]))
    ]
    return {
        "resolved_location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "forecast": forecast,
    }


# Two tools now. The model reads BOTH descriptions and decides which fits.
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the CURRENT weather (right now) for a place by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and optionally region/country."}
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get the weather FORECAST for upcoming days (today through the next week).",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and optionally region/country."},
                "days": {"type": "integer", "description": "How many days ahead, 1 to 7. Defaults to 3."},
            },
            "required": ["location"],
        },
    },
]

TOOL_FUNCTIONS = {"get_weather": get_weather, "get_forecast": get_forecast}


# ---------------------------------------------------------------------------
# THE AGENT LOOP — now operates on a `messages` list owned by the caller,
# so memory survives between questions.
# ---------------------------------------------------------------------------

def run_agent(messages: list) -> str:
    """`messages` already contains the latest user turn. We append to it
    in place, so the caller keeps the full conversation = short-term memory."""
    for step in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Always record the model's turn in memory (NEW: even the final answer,
        # so the next question can "see" what the agent said).
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool call] {block.name}({json.dumps(block.input)})")
            result = TOOL_FUNCTIONS[block.name](**block.input)
            print(f"  [observation] {json.dumps(result)[:200]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: hit the maximum number of steps without finishing."


def main():
    print("Weather agent v2 (has memory + two tools). Type 'quit' to exit.\n")
    conversation = []  # <-- THIS is the short-term memory; it persists.
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        conversation.append({"role": "user", "content": question})
        answer = run_agent(conversation)  # same list every time = memory
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    main()