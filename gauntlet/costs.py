"""Token pricing and USD cost conversion.

Costs are computed from a small per-model price table in USD per 1M tokens.
The prices below are the published Anthropic list prices; **Amazon Bedrock and
other resellers may charge differently**, so treat these as an estimate, not a
billing source of truth. The lookup tolerates the ``anthropic.`` Bedrock prefix
so a Bedrock model id resolves to the same row as its direct-API counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass

from gauntlet.types import Usage


@dataclass(frozen=True)
class Price:
    """List price in USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float


# List prices (USD / 1M tokens). Bedrock and other resellers may differ.
PRICES: dict[str, Price] = {
    "claude-opus-4-8": Price(5.0, 25.0),
    "claude-sonnet-5": Price(3.0, 15.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
}


def _canonical(model: str) -> str:
    """Strip the Bedrock ``anthropic.`` prefix for price-table lookup."""
    return model[len("anthropic.") :] if model.startswith("anthropic.") else model


def price_for(model: str) -> Price | None:
    """Return the price row for a model id, or ``None`` if unknown."""
    return PRICES.get(_canonical(model))


def usd_cost(model: str, usage: Usage) -> float:
    """Convert token usage to an estimated USD cost.

    Returns ``0.0`` for models not in the table (e.g. the scripted provider),
    since there is no meaningful price to apply — callers should check
    :func:`price_for` when a missing price should be surfaced.
    """
    price = price_for(model)
    if price is None:
        return 0.0
    return (
        usage.input_tokens / 1_000_000 * price.input_per_mtok
        + usage.output_tokens / 1_000_000 * price.output_per_mtok
    )
