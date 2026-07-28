"""Harness runs a scripted task end-to-end and records the trajectory."""

from __future__ import annotations

from gauntlet.harness import run_attempt
from gauntlet.providers import ScriptedProvider
from gauntlet.types import Task


def _task(**overrides: object) -> Task:
    base = {
        "id": "t",
        "capability": "c",
        "prompt": "do it",
        "tools": ["read_file", "write_file"],
        "initial_state": {"files": {"a.txt": "src"}},
        "graders": [],
        "scripted": [
            {
                "text": "reading",
                "tool_calls": [
                    {"id": "t1", "name": "read_file", "arguments": {"path": "a.txt"}}
                ],
            },
            {
                "text": "writing",
                "tool_calls": [
                    {
                        "id": "t2",
                        "name": "write_file",
                        "arguments": {"path": "b.txt", "content": "src"},
                    }
                ],
            },
            {"text": "done"},
        ],
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


def test_scripted_attempt_records_trajectory_and_mutates_world() -> None:
    task = _task()
    outcome = run_attempt(task, ScriptedProvider(script=task.scripted))
    traj = outcome.trajectory
    assert traj.tool_names == ["read_file", "write_file"]
    assert traj.final_text == "done"
    assert not traj.hit_iteration_cap
    assert outcome.world.files["b.txt"] == "src"


def test_usage_accumulates_across_turns() -> None:
    task = _task()
    outcome = run_attempt(task, ScriptedProvider(script=task.scripted))
    # Three scripted steps, each 100 in / 25 out by default.
    assert outcome.usage.input_tokens == 300
    assert outcome.usage.output_tokens == 75


def test_latency_is_measured() -> None:
    task = _task()
    outcome = run_attempt(task, ScriptedProvider(script=task.scripted))
    assert outcome.latency_seconds >= 0.0


def test_tool_results_captured_on_turns() -> None:
    task = _task()
    outcome = run_attempt(task, ScriptedProvider(script=task.scripted))
    first_turn = outcome.trajectory.turns[0]
    assert first_turn.tool_calls[0].name == "read_file"
    assert first_turn.tool_results[0].content == "src"
    assert not first_turn.tool_results[0].is_error


def test_iteration_cap_flagged_when_script_never_ends() -> None:
    # A script that always calls a tool and never stops -> cap is hit.
    looping = [
        {
            "text": "again",
            "tool_calls": [
                {"id": "x", "name": "read_file", "arguments": {"path": "a.txt"}}
            ],
            "stop_reason": "tool_use",
        }
    ] * 20
    task = _task(scripted=looping)
    outcome = run_attempt(
        task, ScriptedProvider(script=task.scripted), max_iterations=3
    )
    assert outcome.trajectory.hit_iteration_cap
    assert len(outcome.trajectory.turns) == 3


def test_parallel_tool_calls_all_execute() -> None:
    task = Task(
        id="p",
        capability="parallel",
        prompt="read all",
        tools=["read_file"],
        initial_state={"files": {"a": "1", "b": "2", "c": "3"}},
        graders=[],
        scripted=[
            {
                "text": "parallel",
                "tool_calls": [
                    {"id": "a", "name": "read_file", "arguments": {"path": "a"}},
                    {"id": "b", "name": "read_file", "arguments": {"path": "b"}},
                    {"id": "c", "name": "read_file", "arguments": {"path": "c"}},
                ],
            },
            {"text": "done"},
        ],
    )
    outcome = run_attempt(task, ScriptedProvider(script=task.scripted))
    results = outcome.trajectory.turns[0].tool_results
    assert [r.content for r in results] == ["1", "2", "3"]
