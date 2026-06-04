"""
orchestrator_agent_v3.py — v2 (crash-proof + step cap) plus a thought-process log.

Needs tracer.py in the SAME folder. Every run also writes agent_trace.jsonl:
  THOUGHT  = the model's reasoning that turn
  DECISION = stop_reason  (the line that reveals "ended without acting")
  ACTION   = which agent it delegated to, and the task
  OBSERVE  = what that agent returned
"""

import anthropic
from tracer import log

import weather_agent
import shell_agent
import stock_agent
import pc_builder_agent

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_STEPS = 8            # hard cap on reasoning rounds per request
MAX_RESULT_CHARS = 4000  # trim big sub-agent answers before they re-enter context

SYSTEM = (
    "You are an orchestrator. Delegate each request to the most suitable "
    "agent(s) via the tools, then combine their answers into one clear reply."
)


def ask_weather(task):    return weather_agent.run_agent([{"role": "user", "content": task}])
def ask_shell(task):      return shell_agent.run_agent([{"role": "user", "content": task}])
def ask_stocks(task):     return stock_agent.run_agent([{"role": "user", "content": task}])
def ask_pc_builder(task): return pc_builder_agent.run_agent([{"role": "user", "content": task}])

AGENTS = {
    "ask_weather": ask_weather, "ask_shell": ask_shell,
    "ask_stocks": ask_stocks, "ask_pc_builder": ask_pc_builder,
}


def agent_tool(name, description):
    return {
        "name": name, "description": description,
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "The task for this agent."}},
            "required": ["task"],
        },
    }


TOOLS = [
    agent_tool("ask_weather", "Current weather and forecasts for any place."),
    agent_tool("ask_shell", "Run commands on the user's computer."),
    agent_tool("ask_stocks", "Stock, ETF, and crypto prices and trends."),
    agent_tool("ask_pc_builder", "Spec a PC and find second-hand parts."),
]


def run_tool(name: str, args: dict) -> str:
    """Always returns a string. NEVER raises — a raised tool poisons the loop."""
    try:
        if name not in AGENTS:
            return f"error: unknown agent {name!r}"
        return AGENTS[name](**args)[:MAX_RESULT_CHARS]
    except Exception as e:
        return f"error running {name}: {e}"


def run_agent(messages: list) -> str:
    for _ in range(MAX_STEPS):
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # ---- thought-process log ----
        thought = "".join(b.text for b in resp.content if b.type == "text")
        if thought.strip():
            log("THOUGHT", thought)
        log("DECISION", resp.stop_reason)
        # ------------------------------

        if resp.stop_reason != "tool_use":
            return thought

        results = []
        for b in resp.content:
            if b.type == "tool_use":
                log("ACTION", {b.name: b.input.get("task")})
                result = run_tool(b.name, b.input)
                log("OBSERVE", result)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": result})
        messages.append({"role": "user", "content": results})

    log("DECISION", "hit step limit")
    return "Stopped: hit the step limit."


def main():
    print("Orchestrator v3 (logged). Routes to the right agent(s). Type 'quit' to exit.\n")
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