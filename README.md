# mini-agent

A small, clean, **tool-using AI agent** in Python — provider-agnostic, fully
tested, and runnable with **no API key** thanks to a built-in offline provider.

It shows the pattern behind real AI agents without the noise: a capped
**plan → act → observe → answer** loop, a safe tool registry, and a swappable
LLM backend (OpenAI, Anthropic, or a deterministic mock).

```bash
$ python -m miniagent "what is 21 * 2?"
42

$ python -m miniagent --trace "what is 21 * 2?"
  · action: calc({'expression': '21 * 2'}) -> 42
  · final: 42
42
```

## Why it's built this way

| Decision | Reason |
|---|---|
| **Provider behind a `Protocol`** | The agent depends on an interface, not a vendor. Swap OpenAI ↔ Anthropic ↔ mock without touching the loop. |
| **Offline `MockProvider`** | Tests and demos run free and deterministically — no keys, no network, no flaky CI. |
| **Capped loop (`max_steps`)** | An agent that can't stop is a bug. The step limit is enforced and tested. |
| **Safe `calc` (AST, not `eval`)** | Arithmetic is parsed and evaluated over a whitelist — `__import__(...)` is rejected, not executed. |
| **Tools return errors, never raise** | One bad tool call can't crash the run; the agent sees the error and moves on. |

## Architecture

```
          goal
           │
           ▼
      ┌─────────┐   ACTION: tool args   ┌──────────────┐
      │  Agent  │ ────────────────────► │ ToolRegistry │
      │  loop   │ ◄──────────────────── │  (calc, now, │
      └─────────┘      observation      │  word_count) │
           │                            └──────────────┘
           │ FINAL: answer
           ▼
         answer  (+ full step trace)

   LLMProvider  ←  MockProvider | OpenAIProvider | AnthropicProvider
```

## Install

```bash
git clone https://github.com/aymanghmari/mini-agent.git
cd mini-agent
pip install -e .            # core has zero dependencies
pip install -e ".[openai]"  # optional: real OpenAI backend
pip install -e ".[dev]"     # optional: pytest
```

## Use it as a library

```python
from miniagent import Agent, MockProvider, default_registry

agent = Agent(MockProvider(), default_registry())
print(agent.run("count the words in: hello world").answer)   # -> 2
```

### Add your own tool

```python
from miniagent import default_registry

reg = default_registry()

@reg.tool("reverse", "Reverse a string")
def reverse(text: str) -> str:
    return text[::-1]
```

### Use a real model

```python
from miniagent import Agent, OpenAIProvider, default_registry
# needs OPENAI_API_KEY in the environment + `pip install openai`
agent = Agent(OpenAIProvider(model="gpt-4o-mini"), default_registry())
print(agent.run("What is 144 / 12, and what's the time?").answer)
```

## Tests

```bash
pytest -q        # 11 passing
```

## License

MIT © Ayman Ghmari
