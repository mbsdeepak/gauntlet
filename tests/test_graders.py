"""Each grader family: state, trajectory (required/forbidden/order/args/recover),
and the LLM judge (scripted provider)."""

from __future__ import annotations

from gauntlet.environment import World
from gauntlet.graders import (
    grade_attempt,
    grade_llm_judge,
    grade_state,
    grade_trajectory,
)
from gauntlet.harness import AttemptOutcome
from gauntlet.providers import ScriptedProvider
from gauntlet.types import (
    GraderKind,
    GraderSpec,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
    Usage,
)

# --- state grader ---------------------------------------------------------- #


def test_state_grader_pass() -> None:
    world = World.from_initial_state({"config": {"x": True}})
    g = grade_state(world, {"config.x": True})
    assert g.passed and g.score == 1.0


def test_state_grader_fail_on_mismatch() -> None:
    world = World.from_initial_state({"config": {"x": False}})
    g = grade_state(world, {"config.x": True})
    assert not g.passed and g.score == 0.0


def test_state_grader_fail_on_missing_path() -> None:
    world = World.from_initial_state({"config": {}})
    g = grade_state(world, {"config.missing": 1})
    assert not g.passed
    assert any("missing" in r for r in g.reasons)


# --- trajectory grader ----------------------------------------------------- #


def _traj(names: list[str], **kw: object) -> Trajectory:
    turns = [
        Turn(text="", tool_calls=(ToolCall(id=str(i), name=n, arguments={}),))
        for i, n in enumerate(names)
    ]
    t = Trajectory(turns=turns, final_text="ok")
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_trajectory_required_and_forbidden() -> None:
    traj = _traj(["read_file", "write_file"])
    spec = GraderSpec(
        kind=GraderKind.TRAJECTORY,
        required_tools=["read_file"],
        forbidden_tools=["delete_file"],
    )
    assert grade_trajectory(traj, spec).passed


def test_trajectory_forbidden_tool_called_fails() -> None:
    traj = _traj(["delete_file"])
    spec = GraderSpec(kind=GraderKind.TRAJECTORY, forbidden_tools=["delete_file"])
    assert not grade_trajectory(traj, spec).passed


def test_trajectory_missing_required_fails() -> None:
    traj = _traj(["write_file"])
    spec = GraderSpec(kind=GraderKind.TRAJECTORY, required_tools=["read_file"])
    assert not grade_trajectory(traj, spec).passed


def test_trajectory_ordering_satisfied_and_violated() -> None:
    good = _traj(["read_file", "write_file"])
    bad = _traj(["write_file", "read_file"])
    spec = GraderSpec(kind=GraderKind.TRAJECTORY, ordering=["read_file", "write_file"])
    assert grade_trajectory(good, spec).passed
    assert not grade_trajectory(bad, spec).passed


def test_trajectory_required_args_match() -> None:
    turns = [
        Turn(
            text="",
            tool_calls=(
                ToolCall(
                    id="1", name="write_file", arguments={"path": "b", "content": "x"}
                ),
            ),
        )
    ]
    traj = Trajectory(turns=turns, final_text="ok")
    ok = GraderSpec(
        kind=GraderKind.TRAJECTORY, required_args={"write_file": {"path": "b"}}
    )
    bad = GraderSpec(
        kind=GraderKind.TRAJECTORY, required_args={"write_file": {"path": "wrong"}}
    )
    assert grade_trajectory(traj, ok).passed
    assert not grade_trajectory(traj, bad).passed


def test_trajectory_error_recovery() -> None:
    turns = [
        Turn(
            text="",
            tool_calls=(ToolCall(id="1", name="set_config", arguments={}),),
            tool_results=(ToolResult(tool_use_id="1", content="err", is_error=True),),
        ),
        Turn(text="done"),
    ]
    traj = Trajectory(turns=turns, final_text="done")
    spec = GraderSpec(kind=GraderKind.TRAJECTORY, must_recover_from_error=True)
    assert grade_trajectory(traj, spec).passed


def test_trajectory_recovery_fails_without_error() -> None:
    traj = _traj(["set_config"])
    spec = GraderSpec(kind=GraderKind.TRAJECTORY, must_recover_from_error=True)
    assert not grade_trajectory(traj, spec).passed


# --- llm judge ------------------------------------------------------------- #


def test_llm_judge_skipped_offline() -> None:
    g = grade_llm_judge("anything", "rubric", provider=None)
    assert g.passed
    assert any("skipped" in r for r in g.reasons)


def test_llm_judge_parses_score_and_passes() -> None:
    judge = ScriptedProvider(script=[{"text": "SCORE: 9\nLooks correct."}])
    g = grade_llm_judge("512 MB", "must say 512 MB", provider=judge)
    assert g.passed
    assert g.score == 0.9


def test_llm_judge_low_score_fails() -> None:
    judge = ScriptedProvider(script=[{"text": "SCORE: 2\nWrong."}])
    g = grade_llm_judge("wrong", "rubric", provider=judge)
    assert not g.passed
    assert g.score == 0.2


# --- composite ------------------------------------------------------------- #


def test_grade_attempt_requires_all_graders_pass() -> None:
    world = World.from_initial_state({"config": {"x": True}})
    traj = _traj(["set_config"])
    outcome = AttemptOutcome(traj, world, Usage(), 0.0)
    specs = [
        GraderSpec(kind=GraderKind.STATE, expected_state={"config.x": True}),
        GraderSpec(kind=GraderKind.TRAJECTORY, required_tools=["set_config"]),
    ]
    assert grade_attempt(outcome, specs).passed

    # Flip the trajectory requirement so one grader fails -> composite fails.
    specs[1] = GraderSpec(kind=GraderKind.TRAJECTORY, required_tools=["read_file"])
    assert not grade_attempt(outcome, specs).passed


def test_grade_attempt_no_specs_fails() -> None:
    outcome = AttemptOutcome(Trajectory(), World.from_initial_state({}), Usage(), 0.0)
    assert not grade_attempt(outcome, []).passed
