from miniagent import Agent, MockProvider, default_registry
from miniagent.llm import LLMProvider, Message


def _agent():
    return Agent(MockProvider(), default_registry(), max_steps=5)


def test_agent_uses_calc_tool_and_answers():
    result = _agent().run("please calculate 21 * 2")
    assert result.answer == "42"
    assert result.steps == 2  # one action, one final
    assert any("calc" in line for line in result.trace)


def test_agent_answers_directly_without_a_tool():
    result = _agent().run("hello there")
    assert "help" in result.answer.lower()
    assert result.steps == 1


def test_agent_word_count_roundtrip():
    result = _agent().run("count the words in: the quick brown fox")
    assert result.answer == "4"


class _Runaway:
    """A provider that never says FINAL, to prove the step cap holds."""
    def complete(self, messages: list[Message]) -> str:
        return 'ACTION: now {}'


def test_step_limit_is_enforced():
    agent = Agent(_Runaway(), default_registry(), max_steps=3)
    result = agent.run("loop forever")
    assert result.steps == 3
    assert "step limit" in result.answer.lower()


def test_mock_provider_satisfies_protocol():
    assert isinstance(MockProvider(), LLMProvider)
