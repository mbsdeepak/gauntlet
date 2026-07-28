"""Token-to-USD cost conversion."""

from __future__ import annotations

import pytest

from gauntlet.costs import price_for, usd_cost
from gauntlet.types import Usage


def test_opus_cost_known_value() -> None:
    # 1M input + 1M output on Opus 4.8 = $5 + $25 = $30.
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert usd_cost("claude-opus-4-8", usage) == pytest.approx(30.0)


def test_partial_million_scales_linearly() -> None:
    usage = Usage(input_tokens=500_000, output_tokens=200_000)
    # 0.5*5 + 0.2*25 = 2.5 + 5.0 = 7.5
    assert usd_cost("claude-opus-4-8", usage) == pytest.approx(7.5)


def test_bedrock_prefix_resolves_to_same_price() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0)
    assert usd_cost("anthropic.claude-opus-4-8", usage) == pytest.approx(5.0)
    assert price_for("anthropic.claude-sonnet-5") is not None


def test_unknown_model_is_zero_cost() -> None:
    assert usd_cost("scripted", Usage(input_tokens=999, output_tokens=999)) == 0.0
    assert price_for("scripted") is None


def test_sonnet_and_haiku_prices() -> None:
    m = Usage(1_000_000, 1_000_000)
    assert usd_cost("claude-sonnet-5", m) == pytest.approx(18.0)
    assert usd_cost("claude-haiku-4-5", m) == pytest.approx(6.0)
