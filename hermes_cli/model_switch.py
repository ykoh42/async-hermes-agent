"""Model identities shared by runtime model resolution.

The interactive model picker is intentionally absent from the async library.
This module remains at its upstream path because runtime metadata and startup
validation import these stable, side-effect-free definitions.
"""

from __future__ import annotations

import re
from typing import NamedTuple


_HERMES_MODEL_WARNING = (
    "Nous Research Hermes 3 & 4 models are NOT agentic and are not designed "
    "for use with Hermes Agent. They lack the tool-calling capabilities "
    "required for agent workflows. Consider using an agentic model instead "
    "(Claude, GPT, Gemini, DeepSeek, etc.)."
)

_NOUS_HERMES_NON_AGENTIC_RE = re.compile(
    r"(?:^|[/:])hermes[-_ ]?[34](?:[-_.:]|$)",
    re.IGNORECASE,
)


def is_nous_hermes_non_agentic(model_name: str) -> bool:
    """Return whether *model_name* is a Nous Hermes 3/4 chat model."""
    return bool(model_name and _NOUS_HERMES_NON_AGENTIC_RE.search(model_name))


def _check_hermes_model_warning(model_name: str) -> str:
    """Return the non-agentic warning for Nous Hermes 3/4 chat models."""
    return _HERMES_MODEL_WARNING if is_nous_hermes_non_agentic(model_name) else ""


class ModelIdentity(NamedTuple):
    """Vendor slug and family prefix used for catalog resolution."""

    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelIdentity] = {
    "sonnet": ModelIdentity("anthropic", "claude-sonnet"),
    "opus": ModelIdentity("anthropic", "claude-opus"),
    "haiku": ModelIdentity("anthropic", "claude-haiku"),
    "claude": ModelIdentity("anthropic", "claude"),
    "gpt5": ModelIdentity("openai", "gpt-5"),
    "gpt": ModelIdentity("openai", "gpt"),
    "codex": ModelIdentity("openai", "codex"),
    "o3": ModelIdentity("openai", "o3"),
    "o4": ModelIdentity("openai", "o4"),
    "gemini": ModelIdentity("google", "gemini"),
    "deepseek": ModelIdentity("deepseek", "deepseek-chat"),
    "grok": ModelIdentity("x-ai", "grok"),
    "llama": ModelIdentity("meta-llama", "llama"),
    "qwen": ModelIdentity("qwen", "qwen"),
    "minimax": ModelIdentity("minimax", "minimax"),
    "nemotron": ModelIdentity("nvidia", "nemotron"),
    "kimi": ModelIdentity("moonshotai", "kimi"),
    "glm": ModelIdentity("z-ai", "glm"),
    "step": ModelIdentity("stepfun", "step"),
    "mimo": ModelIdentity("xiaomi", "mimo"),
    "trinity": ModelIdentity("arcee-ai", "trinity"),
}
