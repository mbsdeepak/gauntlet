"""Environment determinism and path resolution."""

from __future__ import annotations

import pytest

from gauntlet.environment import World


def test_from_initial_state_deep_copies() -> None:
    seed = {"files": {"a.txt": "x"}, "tickets": {"T": {"status": "open"}}}
    w1 = World.from_initial_state(seed)
    w1.files["a.txt"] = "mutated"
    w1.tickets["T"]["status"] = "closed"
    # Mutating one world (or its nested dicts) must not touch the seed or a
    # second world built from the same seed.
    w2 = World.from_initial_state(seed)
    assert w2.files["a.txt"] == "x"
    assert w2.tickets["T"]["status"] == "open"
    assert seed["files"]["a.txt"] == "x"


def test_same_seed_same_snapshot() -> None:
    seed = {"config": {"k": 1}, "files": {"f": "y"}}
    assert (
        World.from_initial_state(seed).snapshot()
        == World.from_initial_state(seed).snapshot()
    )


def test_get_path_nested_and_top_level() -> None:
    w = World.from_initial_state(
        {"config": {"mode": "prod"}, "tickets": {"T-1": {"status": "open"}}}
    )
    assert w.get_path("config.mode") == "prod"
    assert w.get_path("tickets.T-1.status") == "open"
    assert w.get_path("files") == {}


def test_get_path_missing_raises() -> None:
    w = World.from_initial_state({"config": {}})
    with pytest.raises(KeyError):
        w.get_path("config.nope")
    with pytest.raises(KeyError):
        w.get_path("bogus.section")
