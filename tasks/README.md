# gauntlet task dataset

Each `*.yaml` file here is one eval task. The set is **versioned** (git-tracked)
and **synthetic**: every environment is a small simulated world, not scraped from
the internet, which keeps the tasks out of model training corpora and reduces
contamination. When you change or add a task, bump the dataset version in the
top-level `README.md` changelog.

## Schema

```yaml
id: <unique string id>                 # required; used for stable ordering
capability: <capability tag>           # required; used for the per-capability report
tags: [<free-form>, ...]               # optional labels
prompt: <the user goal>                # required; sent as the first user message
system: <optional system prompt>       # optional; steers behavior (e.g. grounding)
tools: [<tool_name>, ...]              # required; subset of simulated tools exposed
initial_state:                         # seeds the deterministic world
  files:    {<path>: <contents>, ...}
  tickets:  {<id>: {title, body, status}, ...}
  config:   {<key>: <value>, ...}
inject_error:                          # optional; forces the FIRST call to fail
  tool: <tool_name>
  message: <error text returned to the model>
graders:                               # required; one or more; attempt passes iff ALL pass
  - kind: state | trajectory | llm_judge
    # --- state ---
    expected_state: {<dotted.path>: <expected value>, ...}
    # --- trajectory ---
    required_tools:   [<name>, ...]
    forbidden_tools:  [<name>, ...]
    ordering:         [<name>, ...]           # relative call order that must hold
    required_args:    {<name>: {<k>: <v>}}    # a call to <name> must include these args
    must_recover_from_error: true|false       # saw an error result and still completed
    # --- llm_judge (optional; skipped offline) ---
    rubric: <grading rubric text>
scripted:                              # offline solver trajectory replayed by ScriptedProvider
  - text: <assistant text>
    tool_calls:
      - id: <tool_use id>
        name: <tool_name>
        arguments: {<k>: <v>, ...}
  - text: <final assistant text>       # a step with no tool_calls ends the turn
```

### Dotted state paths

Graders address the world with dotted paths: `config.deploy_enabled`,
`tickets.T-42.status`, `files.reports/outage.md`. See
`gauntlet/environment.py::World.get_path`.

## Available tools

`read_file`, `write_file`, `list_files`, `delete_file` (destructive; needs
`confirm=true`), `search_tickets`, `get_ticket`, `update_ticket`, `get_config`,
`set_config`, `ask_user`. Definitions live in `gauntlet/tools.py`.

## Capabilities covered

| File | Capability | What it tests |
|---|---|---|
| `01_single_tool_read` | single-tool-use | one correct tool call |
| `02_multi_step_sequence` | multi-step-sequencing | read → write in order |
| `03_error_recovery` | error-recovery | recover from an injected tool failure |
| `04_disambiguation` | disambiguation | ask instead of guessing; touch nothing |
| `05_refuse_missing_tool` | refusal-no-tool | don't fabricate a tool that isn't offered |
| `06_parallel_safe` | parallel-safe-tool-use | batch independent reads in one turn |
| `07_state_dependent_branch` | state-dependent-branching | branch on a value read at runtime |
| `08_long_horizon` | long-horizon-multi-tool | 5-step triage to a goal state |
| `09_not_in_context_lookup` | not-in-context-lookup | look it up, don't answer from memory |
| `10_destructive_gating` | destructive-action-gating | confirm before a delete |

## Contamination note

Tasks are hand-authored over simulated services. Because the worlds and goals
are synthetic and never published as "answers", memorization is not a shortcut —
a model must actually plan and call the right tools against the seeded state.
Version the set and prefer adding new tasks over editing old ones so historical
runs stay comparable.
