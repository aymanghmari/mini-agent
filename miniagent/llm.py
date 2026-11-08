"""Provider-agnostic LLM layer.

The agent talks to an ``LLMProvider``. Three implementations ship here:

* ``MockProvider``  – deterministic, offline. Lets the whole agent run with no
  API key, which keeps demos and CI fast and free.
* ``OpenAIProvider`` / ``AnthropicProvider`` – thin adapters, imported lazily so
  the package has no hard dependency on either SDK.

Swapping providers never touches the agent: it only depends on this interface.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Message:
    """A single chat message in the running transcript."""

    role: str   # "system" | "user" | "assistant" | "tool"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a transcript into the model's next line."""

    def complete(self, messages: list[Message]) -> str:
        ...


class MockProvider:
    """A tiny, deterministic stand-in for a real model.

    It speaks the same ``ACTION:`` / ``FINAL:`` protocol the agent expects, so
    the full plan-act-observe loop can be exercised offline. It is intentionally
    simple: enough to demonstrate the architecture, not to be clever.
    """

    _ARITH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")

    def complete(self, messages: list[Message]) -> str:
        goal = next((m.content for m in messages if m.role == "user"), "")
        has_observation = any(m.role == "tool" for m in messages)

        # Second pass: a tool already ran, so answer from what it returned.
        if has_observation:
            observation = next(m.content for m in reversed(messages) if m.role == "tool")
            return f"FINAL: {observation}"

        # First pass: pick one tool based on the goal.
        low = goal.lower()
        if self._ARITH.search(goal):
            expr = self._ARITH.search(goal).group(0)
            return f'ACTION: calc {{"expression": "{expr}"}}'
        if "time" in low or "date" in low:
            return 'ACTION: now {}'
        if "count" in low and "word" in low:
            text = goal.split(":", 1)[-1].strip() or goal
            return f'ACTION: word_count {{"text": "{text}"}}'
        return "FINAL: I can help with calculations, the current time, or word counts."


class OpenAIProvider:
    """Adapter for OpenAI chat models. Requires ``openai`` and an API key."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI  # imported lazily on purpose

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    def complete(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": _map_role(m.role), "content": m.content} for m in messages],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:
    """Adapter for Anthropic (Claude) models. Requires ``anthropic`` + API key."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None):
        import anthropic  # imported lazily on purpose

        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def complete(self, messages: list[Message]) -> str:
        system = " ".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": _map_role(m.role), "content": m.content}
            for m in messages
            if m.role in ("user", "assistant", "tool")
        ]
        resp = self._client.messages.create(
            model=self._model, system=system or None, max_tokens=1024, messages=convo
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def _map_role(role: str) -> str:
    # Tool observations are fed back to the model as user turns.
    return "user" if role == "tool" else role
