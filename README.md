# gauntlet

**An agentic tool-use eval set and a rigorous runner for LLMs.** Each task drops
a model into a deterministic simulated tool environment, gives it a goal, lets it
act through tools in a real agent loop, records the full trajectory, and grades
whether it planned correctly, called the right tools, recovered from injected
errors, and reached the goal state — then reports pass@k with confidence
intervals, cost, and latency, broken down by capability.

## Why this exists / what it demonstrates

The bottleneck in agent evals is rarely a shortage of prompts — it's rigor. This
project is built to show the parts that actually matter when you evaluate models
for tool use and agentic behavior:

- **Deterministic, hermetic environments.** Tools mutate an in-memory simulated
  world (fake filesystem, ticket store, config service). No network, no clock,
  no real side effects. The same task seed always produces the same behavior, so
  results are reproducible and graders can inspect final state exactly.
- **pass@k with the unbiased estimator.** We report the Chen et al. (2021)
  pass@k, not "did any of my runs pass."
- **Wilson score confidence intervals.** Every pass rate is reported with a 95%
  Wilson interval — the correct closed form at small sample sizes, where the
  naive Wald interval falls outside `[0, 1]` and under-covers.
- **Cost and latency tracking.** Token usage is accumulated per attempt and
  converted to USD from a per-model price table; wall-clock latency is measured.
- **Calibrated LLM-as-judge.** Open-ended goals can be scored against a rubric by
  a judge model — and the judge is *skippable offline* so nothing silently
  depends on credentials.
- **Contamination-awareness.** Tasks are synthetic simulated worlds, not scraped
  answers, which reduces train-set leakage. See `tasks/README.md`.
- **A versioned dataset.** Tasks live in git as YAML; prefer adding over editing
  so historical runs stay comparable.

Grading is strict and composable: a task can require the world to reach a goal
state *and* require the right tools in the right order *and* forbid a destructive
tool — the attempt passes only if every grader passes. An attempt that hits the
agent-loop iteration cap is always counted as a failure and logged, never
silently dropped.

## Architecture

```
  task (YAML)                     provider
      │                     ┌───────────────────┐
      ▼                     │ AnthropicProvider  │  Bedrock / direct
 ┌─────────┐   seeds   ┌────┴────┐   (or)         │
 │  World  │◀──────────│ harness │◀───────────────┘
 │ (files, │  mutate   │  agent  │   ScriptedProvider (offline solver, no creds)
 │ tickets,│──────────▶│  loop   │
 │ config) │           └────┬────┘
 └─────────┘                │ records
                            ▼
                      ┌───────────┐
                      │Trajectory │  (tool calls + results + final text)
                      └─────┬─────┘
                            ▼
                      ┌───────────┐   state / trajectory / llm-judge
                      │  graders  │──▶ Grade (passed, score, reasons)
                      └─────┬─────┘
                            ▼
                      ┌───────────┐   pass@k • Wilson CI • cost • latency
                      │  metrics  │──▶ by capability + overall
                      └─────┬─────┘
                            ▼
                      report (Markdown + JSON)
```

## Install

Uses [`uv`](https://docs.astral.sh/uv/). Requires Python 3.11+.

```bash
uv sync --extra dev
```

## Quickstart — offline, no credentials

The offline scripted solver replays a hand-authored trajectory per task, so the
entire pipeline runs end-to-end with **no AWS or Anthropic credentials**:

```bash
uv run gauntlet run --scripted
uv run gauntlet run --scripted --k 5 --json-out report.json
uv run gauntlet report report.json    # re-render Markdown from a saved dump
uv run gauntlet list                  # list the tasks and capabilities
```

## Running a real model (AWS Bedrock)

The only requirement is Bedrock access. Provider and model are configured by
environment; the request surface is kept minimal for Opus 4.8 (no
`temperature`/`top_p`/`top_k`; adaptive thinking is opt-in via `--thinking`).

```bash
export GAUNTLET_PROVIDER=bedrock          # default; use "anthropic" for the direct API
export AWS_REGION=us-east-1
export GAUNTLET_MODEL=claude-opus-4-8      # Bedrock id resolves to anthropic.claude-opus-4-8
uv run gauntlet run --k 5 --json-out report.json
uv run gauntlet run --k 5 --judge          # enable the live LLM-judge grader
```

For the direct API instead of Bedrock:

```bash
export GAUNTLET_PROVIDER=anthropic
uv run gauntlet run --model claude-opus-4-8
```

## Sample report (offline scripted solver — NOT real model numbers)

The snippet below is produced by `gauntlet run --scripted`. It is the
deterministic offline solver, **not** a live model — do not read it as a
leaderboard. The report banner says so too.

```markdown
# gauntlet report

> **Provenance: OFFLINE SCRIPTED SOLVER.** These numbers come from deterministic
> replayed trajectories, not a live model. They demonstrate the harness, not
> model capability. Do not cite as model results.

## Overall
- Model: `scripted-solver`
- Tasks: 10
- Attempts (trials): 30
- Pass rate: 100.0% (95% Wilson CI 88.6%–100.0%)
- Est. cost: $0.0000  _(scripted — synthetic tokens)_

## By capability
| Capability | Pass rate | 95% Wilson CI | Trials |
|---|---|---|---|
| destructive-action-gating | 100.0% | 43.9%–100.0% | 3/3 |
| disambiguation            | 100.0% | 43.9%–100.0% | 3/3 |
| error-recovery            | 100.0% | 43.9%–100.0% | 3/3 |
| ...                       | ...   | ...          | ... |
```

(The scripted solver is authored to pass every task — a test asserts this, which
is how the graders stay self-consistent.)

## Design decisions & intentional non-goals

- **Simulated, not real, environments.** Real filesystems and APIs are
  nondeterministic and unsafe to grade against. Hermetic simulation is the whole
  point — it buys reproducibility and exact state grading. The trade-off is that
  the tool surface is small and stylized; it is not a general computer-use
  harness.
- **Small, curated task set over volume.** Ten tasks spanning distinct
  capabilities, each machine-checkable, beats a thousand fuzzy prompts. Breadth
  of *capability coverage* is the axis that matters here, not count.
- **The provider boundary is a hard seam.** All `anthropic` SDK types live in
  `gauntlet/providers.py`; everything else sees a normalized `AssistantTurn`. This
  is what lets the scripted provider drive the identical harness offline, and
  what would let a second backend slot in without touching graders or metrics.
- **Judge is optional and skippable.** LLM-as-judge is useful for open-ended
  goals but is a liability if it's load-bearing for a run to complete. Offline it
  is skipped and reported as non-blocking.
- **Not included (on purpose):** async/parallel task execution, a web UI,
  persistent result storage/DB, statistical significance testing *between*
  models, and streaming. These are straightforward to add on top of the current
  seams but would add surface area without changing what the project
  demonstrates.

## Companion

Pairs with the companion harness repo **`cogs`**.

## License

MIT © 2026 Deepak.
