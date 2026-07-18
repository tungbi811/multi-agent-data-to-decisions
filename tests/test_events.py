from types import SimpleNamespace

from utils.events import coder_code_from_tool_call, next_or_none, safe_event_type


def test_next_or_none_returns_an_event_without_marking_exhaustion():
    event = object()

    assert next_or_none(iter((event,))) == (event, False)


def test_next_or_none_marks_exhausted_iterators():
    event, exhausted = next_or_none(iter(()))

    assert event is None
    assert exhausted is True


def test_coder_code_from_tool_call_handles_malformed_arguments():
    event = SimpleNamespace(
        content=SimpleNamespace(
            sender="Coder",
            tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments="not-json"))],
        )
    )

    assert coder_code_from_tool_call(event) is None


def test_coder_code_from_tool_call_extracts_code():
    event = SimpleNamespace(
        content=SimpleNamespace(
            sender="Coder",
            tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments='{"code":"print(1)"}'))],
        )
    )

    assert coder_code_from_tool_call(event) == "print(1)"


def test_coder_code_from_tool_call_ignores_non_coder_events():
    event = SimpleNamespace(content=SimpleNamespace(sender="DataScientist"))

    assert coder_code_from_tool_call(event) is None


def test_safe_event_type_maps_unknown_values():
    event = SimpleNamespace(type="future_event")

    assert safe_event_type(event) == "unknown"


def test_safe_event_type_handles_missing_values():
    assert safe_event_type(SimpleNamespace()) == "unknown"


def test_safe_event_type_maps_unhashable_values_to_unknown():
    assert safe_event_type(SimpleNamespace(type=[])) == "unknown"
