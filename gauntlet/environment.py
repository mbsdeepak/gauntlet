"""A deterministic, in-memory simulated world that tools mutate.

This is the crux of gauntlet's rigor: tools are *hermetic*. They never touch a
real filesystem, network, or clock — they read and write this ``World``. Because
a world is fully reconstructed from a task's ``initial_state``, the same seed
always produces the same behavior, and a :class:`~gauntlet.graders.StateGrader`
can inspect the final world to decide pass/fail without any nondeterminism.

The world models three small services that between them cover the tasks in the
suite:

* a fake filesystem (path -> file contents),
* a fake ticket/record store (id -> ticket dict),
* a fake key-value config service (key -> value).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class World:
    """Mutable simulated state shared by a task's tools.

    Construct via :meth:`from_initial_state` so deep-copy isolation is enforced
    — two worlds built from the same seed dict never alias each other.
    """

    files: dict[str, str] = field(default_factory=dict)
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_initial_state(cls, initial_state: dict[str, Any]) -> World:
        """Build a fresh, isolated world from a task's initial-state dict.

        Every nested structure is deep-copied so mutating the returned world
        cannot leak back into the task definition (and re-running a task yields
        an identical starting point).
        """
        state = copy.deepcopy(initial_state or {})
        return cls(
            files=dict(state.get("files", {})),
            tickets={k: dict(v) for k, v in state.get("tickets", {}).items()},
            config=dict(state.get("config", {})),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the world as a plain dict (for grading/debug)."""
        return {
            "files": copy.deepcopy(self.files),
            "tickets": copy.deepcopy(self.tickets),
            "config": copy.deepcopy(self.config),
        }

    def get_path(self, path: str) -> Any:
        """Resolve a dotted state path like ``config.deploy_enabled``.

        The first segment names the section (``files``/``tickets``/``config``).
        Because file paths and config keys legitimately contain dots and slashes
        (``files.reports/outage.md``, ``config.max.size``), keys within
        ``files`` and ``config`` are treated as a single opaque remainder — the
        rest of the path after the section name is *not* split further. Only
        ``tickets`` supports a further ``.<attr>`` step, since ticket ids are
        controlled identifiers without dots.

        A bare section name (``files``) returns the whole section dict. Any
        missing segment raises ``KeyError`` so graders fail loudly rather than
        silently comparing ``None``.
        """
        section, _, remainder = path.partition(".")
        container = getattr(self, section, None)
        if container is None:
            raise KeyError(f"unknown world section: {section!r}")
        if not remainder:
            return container

        if section == "tickets":
            tid, _, attr = remainder.partition(".")
            if tid not in container:
                raise KeyError(f"path not found: {path!r}")
            ticket = container[tid]
            if not attr:
                return ticket
            if attr not in ticket:
                raise KeyError(f"path not found: {path!r}")
            return ticket[attr]

        # files / config: the remainder is a single opaque key.
        if remainder not in container:
            raise KeyError(f"path not found: {path!r}")
        return container[remainder]
