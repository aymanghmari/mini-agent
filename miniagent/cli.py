"""Command-line demo.

    python -m miniagent "what is 21 * 2?"
    python -m miniagent --provider openai "summarize my day"

With no provider flag it uses the offline MockProvider, so it runs with no API
key. ``--trace`` prints each step the agent took.
"""
from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .llm import AnthropicProvider, MockProvider, OpenAIProvider
from .tools import default_registry


def _build_provider(name: str):
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    return MockProvider()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miniagent", description="A tiny tool-using AI agent.")
    parser.add_argument("goal", help="What you want the agent to do")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"],
                        default="mock", help="LLM backend (default: mock, no API key needed)")
    parser.add_argument("--trace", action="store_true", help="Print each step")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args(argv)

    agent = Agent(_build_provider(args.provider), default_registry(), max_steps=args.max_steps)
    result = agent.run(args.goal)

    if args.trace:
        for line in result.trace:
            print(f"  · {line}", file=sys.stderr)
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
