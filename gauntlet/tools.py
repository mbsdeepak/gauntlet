"""Simulated tool implementations bound to a :class:`~gauntlet.environment.World`.

Each tool is a small, deterministic function of ``(world, arguments) -> str``.
Tools are exposed to the model as JSON-schema definitions (``name``,
``description``, ``input_schema``) via :meth:`ToolRegistry.schemas`, and executed
via :meth:`ToolRegistry.execute`.

Error injection
---------------
A task can declare ``inject_error = {"tool": "read_file", "message": "..."}``.
The registry then forces the *first* call to that tool to return an error
result (``is_error=True``) without mutating the world, then behaves normally on
subsequent calls. This exercises a model's ability to recover from a transient
tool failure — a distinct capability from happy-path tool use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gauntlet.environment import World
from gauntlet.types import ToolCall, ToolResult

# A tool body: given the world and parsed arguments, mutate the world and/or
# return a human-readable result string. Raising ToolError yields an error
# tool_result the model can recover from.
ToolBody = Callable[[World, dict[str, Any]], str]


class ToolError(Exception):
    """Raised by a tool body to signal a recoverable, in-domain failure."""


@dataclass(frozen=True)
class ToolDef:
    """A named tool: its JSON schema and its executable body."""

    name: str
    description: str
    input_schema: dict[str, Any]
    body: ToolBody

    def schema(self) -> dict[str, Any]:
        """The provider-facing tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --------------------------------------------------------------------------- #
# Tool bodies. Each is pure w.r.t. its (world, args) inputs.
# --------------------------------------------------------------------------- #


def _read_file(world: World, args: dict[str, Any]) -> str:
    path = args["path"]
    if path not in world.files:
        raise ToolError(f"no such file: {path}")
    return world.files[path]


def _write_file(world: World, args: dict[str, Any]) -> str:
    path = args["path"]
    world.files[path] = args["content"]
    return f"wrote {len(args['content'])} bytes to {path}"


def _list_files(world: World, args: dict[str, Any]) -> str:
    prefix = args.get("prefix", "")
    matches = sorted(p for p in world.files if p.startswith(prefix))
    return "\n".join(matches) if matches else "(no files)"


def _delete_file(world: World, args: dict[str, Any]) -> str:
    path = args["path"]
    if not args.get("confirm", False):
        raise ToolError(
            f"refusing to delete {path}: destructive action requires confirm=true"
        )
    if path not in world.files:
        raise ToolError(f"no such file: {path}")
    del world.files[path]
    return f"deleted {path}"


def _search_tickets(world: World, args: dict[str, Any]) -> str:
    query = args["query"].lower()
    hits = [
        tid
        for tid, t in sorted(world.tickets.items())
        if query in (t.get("title", "") + " " + t.get("body", "")).lower()
    ]
    return "\n".join(hits) if hits else "(no matching tickets)"


def _get_ticket(world: World, args: dict[str, Any]) -> str:
    tid = args["ticket_id"]
    if tid not in world.tickets:
        raise ToolError(f"no such ticket: {tid}")
    t = world.tickets[tid]
    return (
        f"{tid}: status={t.get('status')} title={t.get('title')}\n"
        f"{t.get('body', '')}"
    )


def _update_ticket(world: World, args: dict[str, Any]) -> str:
    tid = args["ticket_id"]
    if tid not in world.tickets:
        raise ToolError(f"no such ticket: {tid}")
    world.tickets[tid]["status"] = args["status"]
    return f"{tid} status set to {args['status']}"


def _get_config(world: World, args: dict[str, Any]) -> str:
    key = args["key"]
    if key not in world.config:
        raise ToolError(f"no such config key: {key}")
    return str(world.config[key])


def _set_config(world: World, args: dict[str, Any]) -> str:
    world.config[args["key"]] = args["value"]
    return f"config {args['key']} = {args['value']}"


def _ask_user(world: World, args: dict[str, Any]) -> str:
    # A no-op clarification channel. Its presence in the trajectory is the
    # signal a disambiguation grader looks for; it never mutates the world.
    return f"(clarifying question recorded: {args['question']})"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_STR = {"type": "string"}

_ALL_TOOLS: dict[str, ToolDef] = {
    t.name: t
    for t in [
        ToolDef(
            "read_file",
            "Read the full contents of a file at the given path.",
            {"type": "object", "properties": {"path": _STR}, "required": ["path"]},
            _read_file,
        ),
        ToolDef(
            "write_file",
            "Create or overwrite a file with the given content.",
            {
                "type": "object",
                "properties": {"path": _STR, "content": _STR},
                "required": ["path", "content"],
            },
            _write_file,
        ),
        ToolDef(
            "list_files",
            "List file paths, optionally filtered by a path prefix.",
            {"type": "object", "properties": {"prefix": _STR}, "required": []},
            _list_files,
        ),
        ToolDef(
            "delete_file",
            "Delete a file. Destructive: requires confirm=true to proceed.",
            {
                "type": "object",
                "properties": {"path": _STR, "confirm": {"type": "boolean"}},
                "required": ["path"],
            },
            _delete_file,
        ),
        ToolDef(
            "search_tickets",
            "Search tickets by a text query; returns matching ticket ids.",
            {"type": "object", "properties": {"query": _STR}, "required": ["query"]},
            _search_tickets,
        ),
        ToolDef(
            "get_ticket",
            "Fetch a single ticket's status, title, and body by id.",
            {
                "type": "object",
                "properties": {"ticket_id": _STR},
                "required": ["ticket_id"],
            },
            _get_ticket,
        ),
        ToolDef(
            "update_ticket",
            "Set a ticket's status (e.g. 'open', 'resolved', 'closed').",
            {
                "type": "object",
                "properties": {"ticket_id": _STR, "status": _STR},
                "required": ["ticket_id", "status"],
            },
            _update_ticket,
        ),
        ToolDef(
            "get_config",
            "Read a configuration value by key.",
            {"type": "object", "properties": {"key": _STR}, "required": ["key"]},
            _get_config,
        ),
        ToolDef(
            "set_config",
            "Set a configuration value by key.",
            {
                "type": "object",
                "properties": {"key": _STR, "value": {}},
                "required": ["key", "value"],
            },
            _set_config,
        ),
        ToolDef(
            "ask_user",
            "Ask the user a clarifying question when the request is ambiguous. "
            "Use this instead of guessing when you cannot safely proceed.",
            {
                "type": "object",
                "properties": {"question": _STR},
                "required": ["question"],
            },
            _ask_user,
        ),
    ]
}


class ToolRegistry:
    """Binds a task's tool subset to a world and enforces error injection.

    The registry is per-attempt: it owns the injected-error bookkeeping (which
    resets each attempt because a new registry and world are constructed).
    """

    def __init__(
        self,
        world: World,
        tool_names: list[str],
        inject_error: dict[str, Any] | None = None,
    ) -> None:
        unknown = [n for n in tool_names if n not in _ALL_TOOLS]
        if unknown:
            raise ValueError(f"unknown tool(s) declared by task: {unknown}")
        self.world = world
        self._tools = {name: _ALL_TOOLS[name] for name in tool_names}
        self._inject_error = inject_error
        self._injected_fired = False

    def schemas(self) -> list[dict[str, Any]]:
        """Provider-facing tool definitions, in declared order."""
        return [t.schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call against the world, honoring error injection.

        A call to a tool the task did not expose is itself an error result (this
        is how the "refuse to use a fabricated tool" negative test bites — the
        model that hallucinates a tool gets an error, not a crash).
        """
        if call.name not in self._tools:
            return ToolResult(
                tool_use_id=call.id,
                content=f"error: tool {call.name!r} is not available for this task",
                is_error=True,
            )

        if (
            self._inject_error is not None
            and not self._injected_fired
            and call.name == self._inject_error.get("tool")
        ):
            self._injected_fired = True
            message = self._inject_error.get("message", "injected transient failure")
            return ToolResult(
                tool_use_id=call.id,
                content=f"error: {message}",
                is_error=True,
            )

        try:
            content = self._tools[call.name].body(self.world, call.arguments)
            return ToolResult(tool_use_id=call.id, content=content, is_error=False)
        except (ToolError, KeyError) as exc:
            return ToolResult(
                tool_use_id=call.id, content=f"error: {exc}", is_error=True
            )


def available_tools() -> list[str]:
    """Names of every tool the framework knows how to simulate."""
    return sorted(_ALL_TOOLS)
