"""Runs one model attempt against one task.

The harness owns the agent loop: it seeds a fresh :class:`World`, binds the
task's tools, and drives the provider through the standard Messages API tool
loop until the model ends its turn (or a bound is hit). It records the full
:class:`~gauntlet.types.Trajectory`, measures wall-clock latency, and
accumulates token usage. It never grades — grading is a separate, pluggable
concern (:mod:`gauntlet.graders`) so the same trajectory can be scored many
ways.

Message construction follows the Messages API contract exactly: on
``stop_reason == "tool_use"`` we append the assistant's content, execute every
tool_use block, then send a *single* user message containing one ``tool_result``
block per call (with ``tool_use_id``, ``content``, ``is_error``), and loop.

The loop is bounded by ``max_iterations``. Hitting the bound is recorded on the
trajectory (``hit_iteration_cap``) and surfaced by the runner — it is never
silently treated as success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from gauntlet.environment import World
from gauntlet.providers import Provider
from gauntlet.tools import ToolRegistry
from gauntlet.types import Task, ToolResult, Trajectory, Turn, Usage


@dataclass
class AttemptOutcome:
    """Everything one attempt produced, before grading."""

    trajectory: Trajectory
    world: World
    usage: Usage
    latency_seconds: float


def _assistant_message(turn_text: str, content_blocks: Any) -> dict[str, Any]:
    """Build the assistant message to echo back on the next request.

    For a real provider we echo the exact ``content`` blocks the API returned
    (required so tool_use ids line up). For the scripted provider there are no
    real blocks, so we synthesize an equivalent content list.
    """
    if content_blocks is not None:
        return {"role": "assistant", "content": content_blocks}
    # Scripted path: reconstruct content blocks from the normalized turn.
    blocks: list[dict[str, Any]] = []
    if turn_text:
        blocks.append({"type": "text", "text": turn_text})
    return {"role": "assistant", "content": blocks}


def _tool_result_blocks(results: list[ToolResult]) -> dict[str, Any]:
    """One user message carrying every tool_result for the turn."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_use_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ],
    }


def run_attempt(
    task: Task,
    provider: Provider,
    max_iterations: int = 10,
) -> AttemptOutcome:
    """Execute a single agent attempt at ``task`` using ``provider``.

    Returns the recorded trajectory, the final world (for state grading), the
    accumulated usage, and the wall-clock latency.
    """
    world = World.from_initial_state(task.initial_state)
    registry = ToolRegistry(world, task.tools, task.inject_error)
    tool_schemas = registry.schemas()

    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    trajectory = Trajectory()
    total_usage = Usage()

    start = time.perf_counter()
    for _ in range(max_iterations):
        assistant = provider.complete(task.system, messages, tool_schemas)
        total_usage = total_usage + assistant.usage

        # Assistant echo for the next request.
        messages.append(
            _assistant_message(assistant.text, assistant.content_blocks)
        )

        if assistant.stop_reason != "tool_use" or not assistant.tool_calls:
            trajectory.turns.append(
                Turn(
                    text=assistant.text,
                    tool_calls=assistant.tool_calls,
                    stop_reason=assistant.stop_reason,
                )
            )
            trajectory.final_text = assistant.text
            break

        # Execute every requested tool call and collect one result each.
        results = [registry.execute(call) for call in assistant.tool_calls]
        trajectory.turns.append(
            Turn(
                text=assistant.text,
                tool_calls=assistant.tool_calls,
                tool_results=tuple(results),
                stop_reason=assistant.stop_reason,
            )
        )
        messages.append(_tool_result_blocks(results))
    else:
        # Loop exhausted without a natural stop.
        trajectory.hit_iteration_cap = True

    latency = time.perf_counter() - start
    return AttemptOutcome(
        trajectory=trajectory,
        world=world,
        usage=total_usage,
        latency_seconds=latency,
    )
