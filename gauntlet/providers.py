"""Model providers: the boundary between gauntlet and the model backend.

A :class:`Provider` takes the conversation-so-far plus the tool schemas and
returns a normalized :class:`AssistantTurn`. Two implementations exist:

* :class:`AnthropicProvider` — talks to Claude via the ``anthropic`` SDK, on
  AWS Bedrock (default) or the direct API. All SDK types are confined to this
  module; the rest of the codebase only ever sees the normalized turn. The SDK
  is imported lazily so the package imports (and the whole test suite runs)
  with no ``anthropic`` install and no credentials.
* :class:`ScriptedProvider` — replays a fixed sequence of assistant turns from
  a task's ``scripted`` field. This is what makes the offline demo and the
  entire test suite hermetic: a scripted "solver" drives the harness end to end
  with zero network and zero credentials.

Opus 4.8 request discipline
---------------------------
On Opus 4.8 we send only ``model``, ``max_tokens``, ``system``, ``messages``,
and ``tools``. ``temperature``/``top_p``/``top_k`` 400 on this model and are
never sent; adaptive thinking is opt-in and off by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from gauntlet.types import ToolCall, Usage

# Provider defaults. Bedrock ids are the anthropic.-prefixed variants; the
# direct API uses the bare id. Only client construction and the id prefix
# differ — the .messages.create(...) surface is identical.
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class AssistantTurn:
    """A provider-normalized assistant response.

    ``content_blocks`` is the raw provider content echoed back verbatim on the
    next request (required by the Messages API tool loop). It is opaque to the
    rest of gauntlet; only the provider that produced it interprets it.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str
    usage: Usage
    content_blocks: Any = None


class Provider(Protocol):
    """The narrow contract the harness depends on."""

    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """Produce the next assistant turn given the conversation and tools."""
        ...


# --------------------------------------------------------------------------- #
# Anthropic provider (Bedrock default / direct)
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK (Bedrock or direct API).

    Configuration is entirely by environment so the same binary runs in either
    mode:

    * ``GAUNTLET_PROVIDER`` — ``bedrock`` (default) or ``anthropic``
    * ``GAUNTLET_MODEL``    — override the model id
    * ``AWS_REGION``        — Bedrock region (default ``us-east-1``)

    The SDK import and client construction are deferred to first use so importing
    this module never requires ``anthropic`` or any credentials.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking: bool = False,
    ) -> None:
        self.provider = provider or os.environ.get("GAUNTLET_PROVIDER", "bedrock")
        self.max_tokens = max_tokens
        self.thinking = thinking
        env_model = os.environ.get("GAUNTLET_MODEL")
        base_model = model or env_model or DEFAULT_MODEL
        # Bedrock ids carry the anthropic. prefix; direct ids do not.
        if self.provider == "bedrock" and not base_model.startswith("anthropic."):
            self.model = f"anthropic.{base_model}"
        elif self.provider == "anthropic" and base_model.startswith("anthropic."):
            self.model = base_model[len("anthropic.") :]
        else:
            self.model = base_model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "bedrock":
            from anthropic import AnthropicBedrock

            self._client = AnthropicBedrock(
                aws_region=os.environ.get("AWS_REGION", "us-east-1")
            )
        else:
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        client = self._get_client()
        # Minimal request surface — see module docstring. No sampling params.
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = tools
        if self.thinking:
            request["thinking"] = {"type": "adaptive"}

        response = client.messages.create(**request)
        return self._normalize(response)

    @staticmethod
    def _normalize(response: Any) -> AssistantTurn:
        """Fold a provider response into an :class:`AssistantTurn`."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason,
            usage=usage,
            content_blocks=response.content,
        )


# --------------------------------------------------------------------------- #
# Scripted provider (offline solver / tests)
# --------------------------------------------------------------------------- #


@dataclass
class _ScriptedStep:
    """One pre-baked assistant turn from a task's ``scripted`` field."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str


@dataclass
class ScriptedProvider:
    """Replays a fixed sequence of assistant turns — no network, no creds.

    Each step in ``script`` is a dict of the form::

        {"text": "...", "stop_reason": "tool_use",
         "tool_calls": [{"id": "t1", "name": "read_file",
                         "arguments": {"path": "a.txt"}}]}

    ``stop_reason`` defaults to ``tool_use`` when the step has tool calls and
    ``end_turn`` otherwise. A synthetic per-step usage is emitted so cost and
    latency plumbing is exercised offline; it is clearly not a real model
    number and the reporter labels scripted runs as such.
    """

    name: str = "scripted"
    script: list[dict[str, Any]] = field(default_factory=list)
    per_step_input_tokens: int = 100
    per_step_output_tokens: int = 25

    def __post_init__(self) -> None:
        self._steps = [self._parse(step) for step in self.script]
        self._cursor = 0

    @staticmethod
    def _parse(step: dict[str, Any]) -> _ScriptedStep:
        calls = tuple(
            ToolCall(
                id=c["id"],
                name=c["name"],
                arguments=dict(c.get("arguments", {})),
            )
            for c in step.get("tool_calls", [])
        )
        default_stop = "tool_use" if calls else "end_turn"
        return _ScriptedStep(
            text=step.get("text", ""),
            tool_calls=calls,
            stop_reason=step.get("stop_reason", default_stop),
        )

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        if self._cursor >= len(self._steps):
            # Script exhausted: end cleanly rather than looping forever.
            return AssistantTurn(
                text="",
                tool_calls=(),
                stop_reason="end_turn",
                usage=Usage(),
            )
        step = self._steps[self._cursor]
        self._cursor += 1
        return AssistantTurn(
            text=step.text,
            tool_calls=step.tool_calls,
            stop_reason=step.stop_reason,
            usage=Usage(
                input_tokens=self.per_step_input_tokens,
                output_tokens=self.per_step_output_tokens,
            ),
            content_blocks=None,
        )
