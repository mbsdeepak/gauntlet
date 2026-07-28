"""End-to-end: run the real scripted task suite and render a report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gauntlet.report import (
    load_suite_json,
    markdown_from_json,
    to_json,
    to_markdown,
)
from gauntlet.runner import load_tasks, run_suite

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@pytest.fixture(scope="module")
def scripted_suite():  # type: ignore[no-untyped-def]
    tasks = load_tasks(TASKS_DIR)
    return tasks, run_suite(tasks, model="scripted-solver", k=3, scripted=True)


def test_all_tasks_load(scripted_suite) -> None:  # type: ignore[no-untyped-def]
    tasks, _ = scripted_suite
    assert len(tasks) == 10
    # every task declares at least one grader and a scripted trajectory
    for t in tasks:
        assert t.graders
        assert t.scripted


def test_scripted_solvers_all_pass(scripted_suite) -> None:  # type: ignore[no-untyped-def]
    """The offline solvers are authored to pass their own tasks — this is the
    fixture that keeps the demo honest and the graders self-consistent."""
    _, suite = scripted_suite
    for r in suite.task_results:
        assert r.num_passed == r.k, (
            f"{r.task_id}: only {r.num_passed}/{r.k} passed; "
            f"reasons: {r.grades[0].reasons}"
        )


def test_suite_is_deterministic() -> None:
    tasks = load_tasks(TASKS_DIR)
    a = to_json(run_suite(tasks, model="m", k=2, scripted=True))
    b = to_json(run_suite(tasks, model="m", k=2, scripted=True))
    # Latency is wall-clock, so strip it before comparing structural output.
    da, db = json.loads(a), json.loads(b)
    for d in (da, db):
        d["overall"].pop("latency_seconds")
        for t in d["tasks"]:
            t.pop("latency_seconds")
    assert da == db


def test_report_labels_scripted_provenance(scripted_suite) -> None:  # type: ignore[no-untyped-def]
    _, suite = scripted_suite
    md = to_markdown(suite)
    assert "OFFLINE SCRIPTED SOLVER" in md
    assert "By capability" in md
    assert "Leaderboard" in md


def test_json_roundtrip_and_rerender(scripted_suite) -> None:  # type: ignore[no-untyped-def]
    _, suite = scripted_suite
    payload = load_suite_json(to_json(suite))
    assert payload["scripted"] is True
    assert payload["overall"]["trials"] == 30  # 10 tasks * k=3
    assert payload["overall"]["passed"] == 30
    md = markdown_from_json(payload)
    assert "gauntlet report" in md
    assert "OFFLINE SCRIPTED SOLVER" in md


def test_every_capability_appears_in_report(scripted_suite) -> None:  # type: ignore[no-untyped-def]
    tasks, suite = scripted_suite
    md = to_markdown(suite)
    for cap in {t.capability for t in tasks}:
        assert cap in md


def test_iteration_cap_counts_as_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A scripted solver that loops forever must be reported as a failure, not
    silently dropped or counted as a pass."""
    # The script always requests a tool and never emits an end_turn, so with a
    # small iteration cap the loop is aborted rather than finishing.
    loop_step = (
        "  - text: again\n"
        "    stop_reason: tool_use\n"
        "    tool_calls: [{id: x, name: read_file, arguments: {path: a}}]\n"
    )
    task_yaml = tmp_path / "loop.yaml"
    task_yaml.write_text(
        "id: loop\n"
        "capability: pathological\n"
        "prompt: loop\n"
        "tools: [read_file]\n"
        "initial_state: {files: {a: '1'}}\n"
        "graders:\n"
        "  - kind: trajectory\n"
        "    required_tools: [read_file]\n"
        "scripted:\n" + loop_step * 5
    )
    tasks = load_tasks(tmp_path)
    suite = run_suite(tasks, model="m", k=1, scripted=True, max_iterations=2)
    result = suite.task_results[0]
    assert result.trajectories[0].hit_iteration_cap
    assert result.num_passed == 0
