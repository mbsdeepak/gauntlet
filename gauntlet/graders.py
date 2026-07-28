"""Pluggable graders that turn an attempt into a :class:`~gauntlet.types.Grade`.

Three families, each answering a different question about an attempt:

* :func:`grade_state` — did the world end up correct? (goal-state predicate)
* :func:`grade_trajectory` — did the model take a valid path? (required/forbidden
  tools, ordering, expected args, error recovery)
* :func:`grade_llm_judge` — is an open-ended answer good against a rubric?
  (optional; uses a provider, and is skippable offline)

A task composes one or more grader specs; :func:`grade_attempt` runs each and
combines them: the attempt passes iff *every* grader passes, and the composite
score is the mean of the component scores. Composing this way keeps the pass@k
signal strict — a task that "reached the goal state but called a forbidden
destructive tool" does not pass.
"""

from __future__ import annotations

from gauntlet.environment import World
from gauntlet.harness import AttemptOutcome
from gauntlet.providers import Provider
from gauntlet.types import Grade, GraderKind, GraderSpec, Trajectory


def grade_state(world: World, expected: dict[str, object]) -> Grade:
    """Assert the final world matches every ``path -> value`` in ``expected``.

    Paths are dotted (see :meth:`World.get_path`). A missing path or a mismatch
    fails the whole grader, with a reason per checked path.
    """
    reasons: list[str] = []
    all_ok = True
    for path, want in expected.items():
        try:
            got = world.get_path(path)
        except KeyError:
            all_ok = False
            reasons.append(f"{path}: missing (expected {want!r})")
            continue
        if got == want:
            reasons.append(f"{path}: ok ({got!r})")
        else:
            all_ok = False
            reasons.append(f"{path}: got {got!r}, expected {want!r}")
    score = 1.0 if all_ok else 0.0
    return Grade(passed=all_ok, score=score, reasons=tuple(reasons))


def grade_trajectory(trajectory: Trajectory, spec: GraderSpec) -> Grade:
    """Check the path the model took against the trajectory constraints.

    Enforces, in order of reasons emitted:

    * every ``required_tools`` name was called at least once,
    * no ``forbidden_tools`` name was called,
    * calls to tools named in ``ordering`` occurred in that relative order,
    * each ``required_args`` tool was called with (at least) those argument
      key/values,
    * if ``must_recover_from_error``, some tool produced an error result and the
      attempt still continued to a natural end (proof of recovery).
    """
    reasons: list[str] = []
    ok = True
    called = trajectory.tool_names

    for name in spec.required_tools:
        if name in called:
            reasons.append(f"required tool {name!r}: called")
        else:
            ok = False
            reasons.append(f"required tool {name!r}: NOT called")

    for name in spec.forbidden_tools:
        if name in called:
            ok = False
            reasons.append(f"forbidden tool {name!r}: called (violation)")
        else:
            reasons.append(f"forbidden tool {name!r}: not called")

    if spec.ordering:
        positions = []
        order_ok = True
        for name in spec.ordering:
            try:
                positions.append(called.index(name))
            except ValueError:
                order_ok = False
                reasons.append(f"ordering: {name!r} never called")
                break
        if order_ok and positions == sorted(positions):
            reasons.append(f"ordering {spec.ordering}: satisfied")
        elif order_ok:
            ok = False
            reasons.append(f"ordering {spec.ordering}: violated")
        else:
            ok = False

    for name, want_args in spec.required_args.items():
        matched = any(
            call.name == name
            and all(call.arguments.get(k) == v for k, v in want_args.items())
            for call in trajectory.tool_calls
        )
        if matched:
            reasons.append(f"args for {name!r}: matched {want_args}")
        else:
            ok = False
            reasons.append(f"args for {name!r}: no call matched {want_args}")

    if spec.must_recover_from_error:
        saw_error = any(
            r.is_error for turn in trajectory.turns for r in turn.tool_results
        )
        recovered = saw_error and not trajectory.hit_iteration_cap
        if recovered:
            reasons.append("error recovery: hit an error and still completed")
        else:
            ok = False
            reasons.append(
                "error recovery: expected an error followed by completion"
            )

    return Grade(passed=ok, score=1.0 if ok else 0.0, reasons=tuple(reasons))


def grade_llm_judge(
    final_text: str,
    rubric: str,
    provider: Provider | None,
) -> Grade:
    """Score open-ended output against ``rubric`` using an LLM judge.

    The judge is asked to return a single line ``SCORE: <0-10>`` plus a short
    justification; the score maps to ``[0, 1]`` and passes at ``>= 0.6``. When
    no provider is supplied (offline / scripted runs) the grader is *skipped*
    and reported as a non-blocking pass so a suite can run without credentials.
    """
    if provider is None:
        return Grade(
            passed=True,
            score=1.0,
            reasons=("llm-judge skipped (no provider available offline)",),
        )

    system = (
        "You are a strict grader. Score the RESPONSE against the RUBRIC on a "
        "0-10 integer scale. Reply with exactly one line 'SCORE: <n>' followed "
        "by one sentence of justification."
    )
    user = f"RUBRIC:\n{rubric}\n\nRESPONSE:\n{final_text}"
    turn = provider.complete(
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[],
    )
    score_10 = _parse_judge_score(turn.text)
    score = score_10 / 10.0
    passed = score >= 0.6
    return Grade(
        passed=passed,
        score=score,
        reasons=(f"llm-judge score {score_10}/10", turn.text.strip()),
    )


def _parse_judge_score(text: str) -> int:
    """Extract the integer after ``SCORE:`` (clamped to 0-10); 0 if absent."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SCORE:"):
            token = stripped.split(":", 1)[1].strip().split()[0]
            try:
                return max(0, min(10, int(float(token))))
            except (ValueError, IndexError):
                return 0
    return 0


def grade_attempt(
    outcome: AttemptOutcome,
    specs: list[GraderSpec],
    judge_provider: Provider | None = None,
) -> Grade:
    """Run every grader spec and combine them into one composite grade.

    Passes iff all component graders pass; composite score is the mean of the
    component scores. Reasons are prefixed with the grader kind for readability.
    """
    # An attempt that hit the harness iteration cap never finished on its own.
    # We treat that as a hard failure regardless of grader specs, so a runaway
    # loop can never be scored as a pass.
    if outcome.trajectory.hit_iteration_cap:
        return Grade(
            passed=False,
            score=0.0,
            reasons=("attempt hit the iteration cap before completing",),
        )

    components: list[Grade] = []
    for spec in specs:
        if spec.kind is GraderKind.STATE:
            components.append(grade_state(outcome.world, spec.expected_state))
        elif spec.kind is GraderKind.TRAJECTORY:
            components.append(grade_trajectory(outcome.trajectory, spec))
        elif spec.kind is GraderKind.LLM_JUDGE:
            components.append(
                grade_llm_judge(
                    outcome.trajectory.final_text, spec.rubric, judge_provider
                )
            )

    if not components:
        return Grade(passed=False, score=0.0, reasons=("no graders defined",))

    passed = all(c.passed for c in components)
    score = sum(c.score for c in components) / len(components)
    reasons: list[str] = []
    for spec, comp in zip(specs, components, strict=True):
        reasons.extend(f"[{spec.kind.value}] {r}" for r in comp.reasons)
    return Grade(passed=passed, score=score, reasons=tuple(reasons))
