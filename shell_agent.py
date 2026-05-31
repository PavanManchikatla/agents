"""
shell_agent_v2.py — now you SEE the model's plan before you approve.

Change from v1: the model explains, in plain English, what it's about to do
(and what each command will change) BEFORE the y/N prompt. So you decide with
full context instead of staring at a bare command.
"""

import json
import subprocess
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# Tell the model to narrate its intent before proposing a command.
SYSTEM = (
    "You can run shell commands on the user's computer. Before each command, "
    "briefly explain in plain English what it does and what it will change. "
    "Prefer safe, reversible commands, and clearly warn before anything "
    "destructive (deleting, overwriting, formatting)."
)


def run_command(command: str) -> dict:
    print(f"\n  proposed: {command}")
    if input("  run it? [y/N] ").strip().lower() != "y":
        return {"status": "declined by user"}
    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
    return {"exit_code": proc.returncode, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:]}


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
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # NEW: show the model's plan for THIS turn before anything is approved or run.
        plan = "".join(b.text for b in resp.content if b.type == "text")
        if plan.strip():
            print(f"\nagent> {plan}")

        if resp.stop_reason != "tool_use":
            return plan

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
    print("Shell agent v2. It explains its plan before each command. Type 'quit' to exit.\n")
    conversation = []
    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            conversation.append({"role": "user", "content": q})
            run_agent(conversation)  # all printing happens inside the loop now
            print()


if __name__ == "__main__":
    main()