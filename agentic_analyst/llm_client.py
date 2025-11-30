"""Utilities for interacting with OpenAI models."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore


load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> Optional["OpenAI"]:
    """Return a cached OpenAI client if the SDK and API key are available."""
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def is_configured() -> bool:
    """True when the OpenAI SDK can be used."""
    return get_client() is not None


def get_model(default: str = "gpt-5") -> str:
    """Return the preferred model name, falling back to a sensible default."""
    return os.getenv("OPENAI_MODEL", default)

