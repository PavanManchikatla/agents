# Agents

A small Python playground for building tool-using AI agents with the Anthropic API. The project includes several standalone agents plus an orchestrator that routes requests to the right specialist.

## Included agents

| File | Purpose |
| --- | --- |
| `Orchestrator_agent.py` | Routes requests to one or more specialist agents and combines their responses. |
| `weather_agent.py` | Retrieves current weather and 1–7 day forecasts from Open-Meteo. |
| `stock_agent.py` | Looks up current and recent stock, ETF, and cryptocurrency prices with Yahoo Finance. |
| `pc_builder_agent.py` | Recommends PC parts and searches the web for second-hand listings. |
| `shell_agent.py` | Proposes shell commands, explains them, and asks for approval before execution. |
| `tracer.py` | Writes orchestrator activity to `agent_trace.jsonl`. |

## Requirements

- Python 3.10 or newer
- An [Anthropic API key](https://console.anthropic.com/)
- Internet access for API calls, weather data, market data, and web search

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install anthropic requests yfinance ddgs
```

If `ddgs` is unavailable in your environment, the PC builder also supports the older package name:

```bash
pip install duckduckgo-search
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## Usage

Run the orchestrator to access all specialist agents from one conversation:

```bash
python Orchestrator_agent.py
```

Or run an agent directly:

```bash
python weather_agent.py
python stock_agent.py
python pc_builder_agent.py
python shell_agent.py
```

Type `quit`, `exit`, or `q` to end an interactive session.

## How it works

Each agent sends the conversation and its tool definitions to the Anthropic Messages API. When the model requests a tool, the Python agent runs the corresponding function, returns the result to the model, and repeats until the model produces a final answer.

The orchestrator exposes each specialist as a tool. It can delegate a request, collect the specialist's result, and synthesize a single response. Its activity is appended to `agent_trace.jsonl` through `tracer.py`.

## Safety notes

- The shell agent can execute commands on your computer. Review every proposed command before approving it.
- Market information is read-only and should not be treated as financial advice.
- Used-part listings and prices may be stale; open each listing to verify price, condition, compatibility, and shipping.
- Trace logs may contain prompts, responses, and tool results. Review them before sharing.
