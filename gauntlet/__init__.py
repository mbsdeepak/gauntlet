"""gauntlet — an agentic tool-use eval set and rigorous runner for LLMs.

Public API surface. The heavy lifting lives in the submodules; this re-exports
the types and functions a library user is most likely to reach for.
"""

from __future__ import annotations

from gauntlet.costs import Price, usd_cost
from gauntlet.environment import World
from gauntlet.graders import grade_attempt, grade_state, grade_trajectory
from gauntlet.harness import run_attempt
from gauntlet.metrics import pass_at_k, wilson_interval
from gauntlet.providers import AnthropicProvider, ScriptedProvider
from gauntlet.runner import load_tasks, run_suite
from gauntlet.types import (
    Grade,
    GraderKind,
    GraderSpec,
    SuiteResult,
    Task,
    TaskResult,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "Grade",
    "GraderKind",
    "GraderSpec",
    "Price",
    "ScriptedProvider",
    "SuiteResult",
    "Task",
    "TaskResult",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "Turn",
    "Usage",
    "World",
    "__version__",
    "grade_attempt",
    "grade_state",
    "grade_trajectory",
    "load_tasks",
    "pass_at_k",
    "run_attempt",
    "run_suite",
    "usd_cost",
    "wilson_interval",
]
