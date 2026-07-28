"""Core data structures for the gauntlet eval framework.

Everything the runner, graders, and reporter pass around is defined here as a
frozen-ish dataclass. Trajectories and results are *records* of what happened;
tasks and grader specs are *declarations* of what should happen. Keeping the two
kinds separate is what lets a task be loaded from YAML, replayed offline, and
graded deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GraderKind(StrEnum):
    """The grader families a task may compose. See :mod:`gauntlet.graders`."""

    STATE = "state"
    TRAJECTORY = "trajectory"
    LLM_JUDGE = "llm_judge"


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the model requested.

    ``id`` mirrors the provider ``tool_use`` block id so results can be matched
    back to calls. ``arguments`` is the parsed JSON input.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing a :class:`ToolCall` against the world."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    """One assistant step in the agent loop.

    A turn carries the assistant's free text, any tool calls it made, and the
    tool results that were fed back. ``stop_reason`` is the provider's reason
    for ending generation (``end_turn``, ``tool_use``, ``max_tokens``, ...).
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    stop_reason: str = "end_turn"


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single attempt.

    Kept provider-agnostic: the provider layer normalizes vendor usage objects
    into this shape so :mod:`gauntlet.costs` never imports the SDK.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class Trajectory:
    """The full record of one agent attempt at one task.

    ``turns`` is the ordered list of assistant steps. ``final_text`` is the last
    assistant text (the model's answer). ``hit_iteration_cap`` records whether
    the harness aborted the loop rather than the model finishing on its own —
    this is surfaced, never silently dropped.
    """

    turns: list[Turn] = field(default_factory=list)
    final_text: str = ""
    hit_iteration_cap: bool = False

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls across all turns, in call order."""
        return [call for turn in self.turns for call in turn.tool_calls]

    @property
    def tool_names(self) -> list[str]:
        """Names of every tool called, in order (may contain duplicates)."""
        return [call.name for call in self.tool_calls]


@dataclass(frozen=True)
class Grade:
    """A grader's verdict for one attempt.

    ``score`` is in ``[0.0, 1.0]``; ``passed`` is the boolean gate used for
    pass@k. ``reasons`` explains the verdict for the report.
    """

    passed: bool
    score: float
    reasons: tuple[str, ...] = ()


@dataclass
class GraderSpec:
    """Declarative grader configuration attached to a task.

    A task composes one or more graders; the attempt passes only if *every*
    grader passes. ``kind`` selects the family; the remaining fields are the
    per-family parameters (only the relevant ones are read).
    """

    kind: GraderKind
    # StateGrader: predicate expressed as expected (path -> value) pairs.
    expected_state: dict[str, Any] = field(default_factory=dict)
    # TrajectoryGrader
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    ordering: list[str] = field(default_factory=list)
    required_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    must_recover_from_error: bool = False
    # LLMJudgeGrader
    rubric: str = ""


@dataclass
class Task:
    """A single eval task: goal, environment seed, tools, and grading.

    ``initial_state`` seeds the deterministic world. ``tools`` names the subset
    of tool implementations exposed to the model. ``inject_error`` optionally
    forces the first call to a named tool to fail (recovery testing).
    ``scripted`` is an offline solver trajectory the :class:`ScriptedProvider`
    replays so the whole pipeline runs with no credentials.
    """

    id: str
    capability: str
    prompt: str
    tools: list[str]
    initial_state: dict[str, Any]
    graders: list[GraderSpec]
    inject_error: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    scripted: list[dict[str, Any]] = field(default_factory=list)
    system: str = ""


@dataclass
class TaskResult:
    """All k attempts at one task by one model, plus aggregate accounting."""

    task_id: str
    capability: str
    model: str
    grades: list[Grade]
    usage: Usage
    latency_seconds: float
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.grades)

    @property
    def num_passed(self) -> int:
        return sum(1 for g in self.grades if g.passed)


@dataclass
class SuiteResult:
    """The outcome of running a whole suite (many tasks, one or more models)."""

    model: str
    provider: str
    scripted: bool
    task_results: list[TaskResult] = field(default_factory=list)
