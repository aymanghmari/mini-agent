"""The agent loop: plan -> act -> observe -> answer.

A ReAct-style loop. Each turn the model returns exactly one line:

    ACTION: <tool_name> <json-args>     # run a tool, feed the result back
    FINAL: <answer>                      # stop and return this to the user

The loop is capped by ``max_steps`` so a misbehaving model can never run
forever. Every step is recorded in ``AgentResult.trace`` for transparency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMProvider, Message
from .tools import ToolRegistry

_SYSTEM = """You are a precise tool-using assistant.
Reply with EXACTLY one line, either:
  ACTION: <tool> <json-args>
  FINAL: <answer>
Available tools:
{tools}
Use a tool when it helps; otherwise answer directly with FINAL."""

_ACTION_RE = re.compile(r"^ACTION:\s*(\w+)\s*(\{.*\})?\s*$", re.DOTALL)
_FINAL_RE = re.compile(r"^FINAL:\s*(.*)$", re.DOTALL)


@dataclass
class AgentResult:
    answer: str
    steps: int
    trace: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry, max_steps: int = 5):
        self._llm = provider
        self._tools = registry
        self._max_steps = max_steps

    def run(self, goal: str) -> AgentResult:
        messages = [
            Message("system", _SYSTEM.format(tools=self._tools.describe())),
            Message("user", goal),
        ]
        trace: list[str] = []

        for step in range(1, self._max_steps + 1):
            reply = self._llm.complete(messages).strip()

            final = _FINAL_RE.match(reply)
            if final:
                trace.append(f"final: {final.group(1).strip()}")
                return AgentResult(final.group(1).strip(), step, trace)

            action = _ACTION_RE.match(reply)
            if not action:
                # Model didn't follow the protocol — return what it said.
                trace.append(f"unparsed: {reply}")
                return AgentResult(reply, step, trace)

            tool = action.group(1)
            args = json.loads(action.group(2)) if action.group(2) else {}
            observation = self._tools.run(tool, args)
            trace.append(f"action: {tool}({args}) -> {observation}")

            messages.append(Message("assistant", reply))
            messages.append(Message("tool", observation))

        return AgentResult("Stopped: reached step limit without a final answer.",
                           self._max_steps, trace)
