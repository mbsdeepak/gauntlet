"""Render a :class:`~gauntlet.types.SuiteResult` to Markdown and JSON.

The Markdown report leads with a provenance banner: every table is explicitly
labeled as coming from real model calls or the offline scripted solver, so a
reader can never mistake a demo run for a leaderboard. Below the banner sit a
per-capability breakdown (with Wilson intervals) and an overall summary line
with cost and latency.

The JSON dump is the machine-readable counterpart — enough to reconstruct the
headline numbers and re-render the Markdown, round-trippable via
:func:`load_suite_json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from gauntlet.costs import usd_cost
from gauntlet.metrics import wilson_interval
from gauntlet.types import SuiteResult


def _provenance_banner(suite: SuiteResult) -> str:
    if suite.scripted:
        return (
            "> **Provenance: OFFLINE SCRIPTED SOLVER.** These numbers come from "
            "deterministic replayed trajectories, not a live model. They "
            "demonstrate the harness, not model capability. Do not cite as "
            "model results."
        )
    return (
        f"> **Provenance: LIVE MODEL** (`{suite.model}` via `{suite.provider}`). "
        "Numbers reflect real API calls."
    )


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(suite: SuiteResult) -> str:
    """Render a full Markdown report for one suite result."""
    lines: list[str] = []
    lines.append("# gauntlet report")
    lines.append("")
    lines.append(_provenance_banner(suite))
    lines.append("")

    total_trials = sum(r.k for r in suite.task_results)
    total_passed = sum(r.num_passed for r in suite.task_results)
    total_usage = _sum_usage(suite)
    total_latency = sum(r.latency_seconds for r in suite.task_results)
    cost = usd_cost(suite.model, total_usage)
    overall_lo, overall_hi = wilson_interval(total_passed, total_trials)

    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Model: `{suite.model}`")
    lines.append(f"- Provider: `{suite.provider}`")
    lines.append(f"- Tasks: {len(suite.task_results)}")
    lines.append(f"- Attempts (trials): {total_trials}")
    lines.append(
        f"- Pass rate: {_fmt_pct(total_passed / total_trials if total_trials else 0)} "
        f"(95% Wilson CI {_fmt_pct(overall_lo)}–{_fmt_pct(overall_hi)})"
    )
    lines.append(
        f"- Tokens: {total_usage.input_tokens} in / "
        f"{total_usage.output_tokens} out"
    )
    lines.append(
        f"- Est. cost: ${cost:.4f}"
        + ("  _(scripted — synthetic tokens)_" if suite.scripted else "")
    )
    lines.append(f"- Total wall-clock: {total_latency:.3f}s")
    lines.append("")

    lines.append("## By capability")
    lines.append("")
    lines.append("| Capability | Pass rate | 95% Wilson CI | Trials |")
    lines.append("|---|---|---|---|")
    for cap, (passed, trials) in sorted(_by_capability(suite).items()):
        lo, hi = wilson_interval(passed, trials)
        lines.append(
            f"| {cap} | {_fmt_pct(passed / trials if trials else 0)} "
            f"| {_fmt_pct(lo)}–{_fmt_pct(hi)} | {passed}/{trials} |"
        )
    lines.append("")

    lines.append("## Leaderboard (per task)")
    lines.append("")
    lines.append("| Task | Capability | Passed | Est. cost | Latency |")
    lines.append("|---|---|---|---|---|")
    for r in suite.task_results:
        lines.append(
            f"| {r.task_id} | {r.capability} | {r.num_passed}/{r.k} "
            f"| ${usd_cost(r.model, r.usage):.4f} | {r.latency_seconds:.3f}s |"
        )
    lines.append("")
    return "\n".join(lines)


def _by_capability(suite: SuiteResult) -> dict[str, tuple[int, int]]:
    """Aggregate (passed, trials) per capability tag."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in suite.task_results:
        agg[r.capability][0] += r.num_passed
        agg[r.capability][1] += r.k
    return {cap: (p, t) for cap, (p, t) in agg.items()}


def _sum_usage(suite: SuiteResult) -> Any:
    from gauntlet.types import Usage

    total = Usage()
    for r in suite.task_results:
        total = total + r.usage
    return total


def to_json(suite: SuiteResult) -> str:
    """Serialize a suite result to a stable, indented JSON string."""
    total_usage = _sum_usage(suite)
    payload = {
        "model": suite.model,
        "provider": suite.provider,
        "scripted": suite.scripted,
        "overall": {
            "trials": sum(r.k for r in suite.task_results),
            "passed": sum(r.num_passed for r in suite.task_results),
            "input_tokens": total_usage.input_tokens,
            "output_tokens": total_usage.output_tokens,
            "est_cost_usd": usd_cost(suite.model, total_usage),
            "latency_seconds": sum(r.latency_seconds for r in suite.task_results),
        },
        "by_capability": {
            cap: {"passed": p, "trials": t}
            for cap, (p, t) in _by_capability(suite).items()
        },
        "tasks": [
            {
                "task_id": r.task_id,
                "capability": r.capability,
                "k": r.k,
                "passed": r.num_passed,
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "est_cost_usd": usd_cost(r.model, r.usage),
                "latency_seconds": r.latency_seconds,
                "grades": [
                    {"passed": g.passed, "score": g.score, "reasons": list(g.reasons)}
                    for g in r.grades
                ],
            }
            for r in suite.task_results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def load_suite_json(text: str) -> dict[str, Any]:
    """Load a JSON dump produced by :func:`to_json`."""
    return json.loads(text)


def markdown_from_json(payload: dict[str, Any]) -> str:
    """Re-render a Markdown report from a loaded JSON payload.

    Used by ``gauntlet report <file.json>`` so a saved run can be re-rendered
    without re-running the suite.
    """
    scripted = payload.get("scripted", False)
    lines = ["# gauntlet report", ""]
    if scripted:
        lines.append(
            "> **Provenance: OFFLINE SCRIPTED SOLVER.** Deterministic replay, "
            "not a live model."
        )
    else:
        lines.append(
            f"> **Provenance: LIVE MODEL** (`{payload['model']}` via "
            f"`{payload['provider']}`)."
        )
    lines.append("")
    overall = payload["overall"]
    trials = overall["trials"]
    passed = overall["passed"]
    lo, hi = wilson_interval(passed, trials)
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Model: `{payload['model']}`")
    lines.append(
        f"- Pass rate: {_fmt_pct(passed / trials if trials else 0)} "
        f"(95% Wilson CI {_fmt_pct(lo)}–{_fmt_pct(hi)})"
    )
    lines.append(f"- Est. cost: ${overall['est_cost_usd']:.4f}")
    lines.append("")
    lines.append("## By capability")
    lines.append("")
    lines.append("| Capability | Pass rate | 95% Wilson CI | Trials |")
    lines.append("|---|---|---|---|")
    for cap, stat in sorted(payload["by_capability"].items()):
        p, t = stat["passed"], stat["trials"]
        clo, chi = wilson_interval(p, t)
        lines.append(
            f"| {cap} | {_fmt_pct(p / t if t else 0)} "
            f"| {_fmt_pct(clo)}–{_fmt_pct(chi)} | {p}/{t} |"
        )
    lines.append("")
    return "\n".join(lines)
