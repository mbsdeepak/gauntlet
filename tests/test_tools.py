"""Tool execution, error handling, and error injection."""

from __future__ import annotations

from gauntlet.environment import World
from gauntlet.tools import ToolRegistry, available_tools
from gauntlet.types import ToolCall


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(id="c", name=name, arguments=dict(args))


def test_write_then_read_roundtrip() -> None:
    world = World.from_initial_state({})
    reg = ToolRegistry(world, ["write_file", "read_file"])
    reg.execute(_call("write_file", path="a.txt", content="hello"))
    result = reg.execute(_call("read_file", path="a.txt"))
    assert not result.is_error
    assert result.content == "hello"


def test_read_missing_file_is_error() -> None:
    reg = ToolRegistry(World.from_initial_state({}), ["read_file"])
    result = reg.execute(_call("read_file", path="nope"))
    assert result.is_error
    assert "no such file" in result.content


def test_delete_requires_confirmation() -> None:
    world = World.from_initial_state({"files": {"x": "v"}})
    reg = ToolRegistry(world, ["delete_file"])
    blocked = reg.execute(_call("delete_file", path="x"))
    assert blocked.is_error and "confirm" in blocked.content
    assert "x" in world.files  # not deleted
    ok = reg.execute(_call("delete_file", path="x", confirm=True))
    assert not ok.is_error
    assert "x" not in world.files


def test_calling_unavailable_tool_is_error_not_crash() -> None:
    reg = ToolRegistry(World.from_initial_state({}), ["read_file"])
    result = reg.execute(_call("send_email", to="a@b.c"))
    assert result.is_error
    assert "not available" in result.content


def test_error_injection_fires_once_then_recovers() -> None:
    world = World.from_initial_state({"config": {"k": False}})
    reg = ToolRegistry(
        world,
        ["set_config"],
        inject_error={"tool": "set_config", "message": "503"},
    )
    first = reg.execute(_call("set_config", key="k", value=True))
    assert first.is_error and "503" in first.content
    assert world.config["k"] is False  # injected failure did not mutate
    second = reg.execute(_call("set_config", key="k", value=True))
    assert not second.is_error
    assert world.config["k"] is True


def test_search_tickets_matches_title_and_body() -> None:
    world = World.from_initial_state(
        {"tickets": {"T-1": {"title": "login broke", "body": "500"}}}
    )
    reg = ToolRegistry(world, ["search_tickets"])
    assert "T-1" in reg.execute(_call("search_tickets", query="login")).content
    assert "no matching" in reg.execute(
        _call("search_tickets", query="zzz")
    ).content


def test_update_ticket_mutates_status() -> None:
    world = World.from_initial_state({"tickets": {"T-1": {"status": "open"}}})
    reg = ToolRegistry(world, ["update_ticket"])
    reg.execute(_call("update_ticket", ticket_id="T-1", status="resolved"))
    assert world.tickets["T-1"]["status"] == "resolved"


def test_schemas_expose_declared_tools_only() -> None:
    reg = ToolRegistry(World.from_initial_state({}), ["read_file", "write_file"])
    names = {s["name"] for s in reg.schemas()}
    assert names == {"read_file", "write_file"}
    for s in reg.schemas():
        assert set(s) == {"name", "description", "input_schema"}


def test_available_tools_covers_registry() -> None:
    assert "ask_user" in available_tools()
    assert len(available_tools()) >= 10
