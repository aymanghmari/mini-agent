"""mini-agent — a small, clean, tool-using AI agent in Python.

Provider-agnostic (OpenAI, Anthropic, or an offline mock), with a safe tool
registry and a capped plan-act-observe loop.
"""
from .agent import Agent, AgentResult
from .llm import AnthropicProvider, LLMProvider, Message, MockProvider, OpenAIProvider
from .tools import Tool, ToolRegistry, default_registry

__version__ = "0.1.0"
__all__ = [
    "Agent", "AgentResult",
    "LLMProvider", "Message", "MockProvider", "OpenAIProvider", "AnthropicProvider",
    "Tool", "ToolRegistry", "default_registry",
]
