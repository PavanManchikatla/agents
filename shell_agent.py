"""
shell_agent.py — an agent that can use your terminal (with your approval).

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python shell_agent.py

Same loop as the weather agent; the only new thing is the tool.
Safety: the model can only PROPOSE a command. It runs only if you type 'y'.
"""

import json
import subprocess
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def run_command(command: str) -> dict:
    # GUARDRAIL: human-in-the-loop. Nothing executes without your 'y'.
    print(f"\n  proposed: {command}")
    if input("  run it? [y/N] ").strip().lower() != "y":
        return {"status": "declined by user"}

    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-3000:],   # trim so huge output can't flood the context
        "stderr": proc.stderr[-3000:],
    }


TOOLS = [{
    "name": "run_command",
    "description": "Run a shell command on the user's computer and return its output.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The shell command to run."}},
        "required": ["command"],
    },
}]


def run_agent(messages: list) -> str:
    while True:
        resp = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for b in resp.content:
            if b.type == "tool_use":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(run_command(**b.input)),
                })
        messages.append({"role": "user", "content": results})


def main():
    print("Shell agent. Ask it to do things in your terminal. Type 'quit' to exit.\n")
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