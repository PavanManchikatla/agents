"""
tracer.py — a tiny thought-process logger you can drop into any agent.

Call log() at each step of the loop. It prints a readable line AND appends a
full JSON record to agent_trace.jsonl, so you can replay any run afterward.

(Named 'tracer', not 'trace', because Python already has a stdlib 'trace'.)
"""

import json
import time

TRACE_FILE = "agent_trace.jsonl"


def log(kind: str, data) -> None:
    """kind is one of THOUGHT / DECISION / ACTION / OBSERVE (any label works)."""
    entry = {"time": time.strftime("%H:%M:%S"), "kind": kind, "data": data}

    shown = str(data)
    if len(shown) > 200:               # keep the console readable...
        shown = shown[:200] + "…"
    print(f"  · {kind:<9}{shown}")

    with open(TRACE_FILE, "a") as f:   # ...but save the full record to disk
        f.write(json.dumps(entry, default=str) + "\n")