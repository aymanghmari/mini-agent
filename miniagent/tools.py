"""The tool layer.

A ``Tool`` is just a named function plus a description. ``ToolRegistry`` holds
them and runs one by name with JSON arguments. Tools are deliberately small,
offline, and side-effect-free so the demo is safe to run anywhere.

Adding a tool is one decorator:

    @registry.tool("greet", "Say hello to someone")
    def greet(name: str) -> str:
        return f"Hello, {name}!"
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, name: str, description: str):
        """Decorator that registers a function as a tool."""
        def wrap(func: Callable[..., str]) -> Callable[..., str]:
            self._tools[name] = Tool(name, description, func)
            return func
        return wrap

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"error: unknown tool '{name}'"
        try:
            return str(self._tools[name].func(**args))
        except Exception as exc:  # tools must never crash the agent loop
            return f"error: {exc}"


# ── A safe arithmetic evaluator (no eval of arbitrary code) ──────────────
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg, ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def default_registry() -> ToolRegistry:
    """A registry with a few harmless, useful demo tools."""
    reg = ToolRegistry()

    @reg.tool("calc", "Evaluate a basic arithmetic expression, e.g. '2 + 3 * 4'")
    def calc(expression: str) -> str:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))

    @reg.tool("now", "Return the current UTC date and time (ISO 8601)")
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @reg.tool("word_count", "Count the words in a piece of text")
    def word_count(text: str) -> str:
        return str(len(text.split()))

    return reg
