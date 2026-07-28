"""Suite orchestration: load tasks, run k attempts each, grade, aggregate.

The runner is the top-level entry point that ties every other module together.
It loads tasks from YAML, and for each task runs ``k`` attempts through the
harness, grades each attempt, and rolls the grades up into a
:class:`~gauntlet.types.SuiteResult`.

Determinism and honesty are the two invariants:

* Tasks are processed in a stable, sorted order and attempts are numbered, so a
  scripted run is byte-for-byte reproducible.
* Any attempt that hit the harness iteration cap is logged at WARNING and
  counted as a *failure* — never silently dropped or counted as a pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from gauntlet.graders import grade_attempt
from gauntlet.harness import run_attempt
from gauntlet.providers import Provider, ScriptedProvider
from gauntlet.types import (
    GraderKind,
    GraderSpec,
    SuiteResult,
    Task,
    TaskResult,
    Usage,
)

logger = logging.getLogger("gauntlet.runner")


def _grader_from_dict(raw: dict[str, Any]) -> GraderSpec:
    """Parse one grader entry from a task YAML into a :class:`GraderSpec`."""
    return GraderSpec(
        kind=GraderKind(raw["kind"]),
        expected_state=raw.get("expected_state", {}),
        required_tools=raw.get("required_tools", []),
        forbidden_tools=raw.get("forbidden_tools", []),
        ordering=raw.get("ordering", []),
        required_args=raw.get("required_args", {}),
        must_recover_from_error=raw.get("must_recover_from_error", False),
        rubric=raw.get("rubric", ""),
    )


def load_task(path: Path) -> Task:
    """Load and validate a single task YAML file into a :class:`Task`."""
    raw = yaml.safe_load(path.read_text())
    return Task(
        id=raw["id"],
        capability=raw["capability"],
        prompt=raw["prompt"],
        tools=raw["tools"],
        initial_state=raw.get("initial_state", {}),
        graders=[_grader_from_dict(g) for g in raw["graders"]],
        inject_error=raw.get("inject_error"),
        tags=raw.get("tags", []),
        scripted=raw.get("scripted", []),
        system=raw.get("system", ""),
    )


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Load every ``*.yaml`` task under ``tasks_dir``, sorted by filename."""
    files = sorted(tasks_dir.glob("*.yaml"))
    return [load_task(f) for f in files]


def run_task(
    task: Task,
    provider_factory: Any,
    model: str,
    k: int,
    max_iterations: int,
    judge_provider: Provider | None,
) -> TaskResult:
    """Run ``k`` attempts of one task and grade each.

    ``provider_factory`` is a callable returning a fresh provider per attempt —
    important for the scripted provider, whose cursor is stateful and must be
    reset each attempt.
    """
    grades = []
    trajectories = []
    total_usage = Usage()
    total_latency = 0.0

    for attempt in range(k):
        provider = provider_factory(task)
        outcome = run_attempt(task, provider, max_iterations=max_iterations)
        if outcome.trajectory.hit_iteration_cap:
            logger.warning(
                "task %s attempt %d/%d hit the %d-iteration cap; counted as fail",
                task.id,
                attempt + 1,
                k,
                max_iterations,
            )
        grade = grade_attempt(outcome, task.graders, judge_provider=judge_provider)
        grades.append(grade)
        trajectories.append(outcome.trajectory)
        total_usage = total_usage + outcome.usage
        total_latency += outcome.latency_seconds

    return TaskResult(
        task_id=task.id,
        capability=task.capability,
        model=model,
        grades=grades,
        usage=total_usage,
        latency_seconds=total_latency,
        trajectories=trajectories,
    )


def run_suite(
    tasks: list[Task],
    model: str,
    k: int = 3,
    scripted: bool = False,
    max_iterations: int = 10,
    judge_provider: Provider | None = None,
    live_provider: Provider | None = None,
) -> SuiteResult:
    """Run a whole suite and aggregate into a :class:`SuiteResult`.

    When ``scripted`` is true, each task is driven by a fresh
    :class:`ScriptedProvider` replaying that task's ``scripted`` field — the
    offline path (no creds). Otherwise ``live_provider`` is used for every
    attempt (a real model). ``judge_provider`` powers any LLM-judge grader; pass
    ``None`` (the default, and the only option offline) to skip judging.
    """
    if scripted:

        def factory(task: Task) -> Provider:
            return ScriptedProvider(script=task.scripted)

        provider_name = "scripted"
    else:
        if live_provider is None:
            raise ValueError("a live_provider is required when scripted=False")

        def factory(task: Task) -> Provider:
            return live_provider

        provider_name = getattr(live_provider, "name", "anthropic")

    results = [
        run_task(task, factory, model, k, max_iterations, judge_provider)
        for task in sorted(tasks, key=lambda t: t.id)
    ]
    return SuiteResult(
        model=model,
        provider=provider_name,
        scripted=scripted,
        task_results=results,
    )
