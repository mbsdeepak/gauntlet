"""Command-line interface for gauntlet.

Three subcommands:

* ``gauntlet run``    — run a suite (offline ``--scripted`` or a real model) and
  emit a Markdown report plus an optional JSON dump.
* ``gauntlet report`` — re-render the Markdown report from a saved JSON dump.
* ``gauntlet list``   — list the tasks in a directory.

``--scripted`` is the credential-free path: it drives every task with its
offline solver trajectory, so ``gauntlet run --scripted`` works with no AWS or
Anthropic credentials at all.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gauntlet import __version__
from gauntlet.providers import AnthropicProvider
from gauntlet.report import load_suite_json, markdown_from_json, to_json, to_markdown
from gauntlet.runner import load_tasks, run_suite

_DEFAULT_TASKS = Path(__file__).resolve().parent.parent / "tasks"


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tasks_dir = Path(args.tasks)
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print(f"no tasks found in {tasks_dir}", file=sys.stderr)
        return 1

    if args.scripted:
        suite = run_suite(
            tasks,
            model=args.model,
            k=args.k,
            scripted=True,
            max_iterations=args.max_iterations,
        )
    else:
        provider = AnthropicProvider(model=args.model, thinking=args.thinking)
        judge = AnthropicProvider(model=args.model) if args.judge else None
        suite = run_suite(
            tasks,
            model=provider.model,
            k=args.k,
            scripted=False,
            max_iterations=args.max_iterations,
            live_provider=provider,
            judge_provider=judge,
        )

    print(to_markdown(suite))
    if args.json_out:
        Path(args.json_out).write_text(to_json(suite))
        print(f"\n(wrote JSON dump to {args.json_out})", file=sys.stderr)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    payload = load_suite_json(Path(args.suiteresult).read_text())
    print(markdown_from_json(payload))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks))
    for t in tasks:
        tag_str = f" [{', '.join(t.tags)}]" if t.tags else ""
        print(f"{t.id:32s} {t.capability:22s}{tag_str}")
    print(f"\n{len(tasks)} tasks", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser (exposed for testing)."""
    parser = argparse.ArgumentParser(prog="gauntlet", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"gauntlet {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an eval suite")
    run.add_argument("--model", default="claude-opus-4-8", help="model id")
    run.add_argument("--k", type=int, default=3, help="attempts per task")
    run.add_argument("--tasks", default=str(_DEFAULT_TASKS), help="tasks directory")
    run.add_argument(
        "--scripted",
        action="store_true",
        help="offline scripted solver (no credentials required)",
    )
    run.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        dest="max_iterations",
        help="max agent-loop iterations per attempt",
    )
    run.add_argument(
        "--thinking",
        action="store_true",
        help="enable adaptive thinking on the live model (off by default)",
    )
    run.add_argument(
        "--judge",
        action="store_true",
        help="use a live LLM judge for llm_judge graders (real model only)",
    )
    run.add_argument("--json-out", dest="json_out", help="also write a JSON dump")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="re-render a report from a JSON dump")
    rep.add_argument("suiteresult", help="path to a suiteresult.json")
    rep.set_defaults(func=_cmd_report)

    lst = sub.add_parser("list", help="list tasks")
    lst.add_argument("--tasks", default=str(_DEFAULT_TASKS), help="tasks directory")
    lst.set_defaults(func=_cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
