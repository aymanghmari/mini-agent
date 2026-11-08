from miniagent.tools import default_registry


def test_calc_respects_precedence():
    reg = default_registry()
    assert reg.run("calc", {"expression": "2 + 3 * 4"}) == "14"


def test_calc_handles_parens_and_unary():
    reg = default_registry()
    assert reg.run("calc", {"expression": "-(2 + 3) * 2"}) == "-10"
    assert reg.run("calc", {"expression": "7 / 2"}) == "3.5"


def test_calc_rejects_code_injection():
    reg = default_registry()
    out = reg.run("calc", {"expression": "__import__('os').system('echo hi')"})
    assert out.startswith("error:")


def test_word_count():
    reg = default_registry()
    assert reg.run("word_count", {"text": "one two three"}) == "3"


def test_unknown_tool_is_reported_not_raised():
    reg = default_registry()
    assert reg.run("nope", {}) == "error: unknown tool 'nope'"


def test_now_is_iso():
    reg = default_registry()
    out = reg.run("now", {})
    assert out.startswith("20") and "T" in out
