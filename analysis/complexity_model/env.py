"""API key loading — spec §2.2. Reads keys from the process environment, optionally
loaded from a `.env` file first (see `.env.example` at the repo root). Only imported by
`run_pilot.py` and the elicitation scripts — never by `capability_model.py`, so the
shipped router still needs zero environment setup.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Slots checked for OpenAI keys, in order. Multiple keys because a single org key can
#: hit per-key rate limits or run out of quota mid-run; `clients.OpenAIClient` rotates
#: across whatever is found here on 401/403/429 (see its docstring).
_OPENAI_KEY_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3")

_ANTHROPIC_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env(env_path: str | Path | None = None) -> None:
    """Load a `.env` file into `os.environ`, if `python-dotenv` is installed and the file
    exists. Never raises — keys can also just be exported in the shell, and a missing
    `.env` is not an error."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path or (REPO_ROOT / ".env"))


def load_keys(env_path: str | Path | None = None) -> tuple[str | None, list[str]]:
    """`(anthropic_key, openai_keys)` — `openai_keys` is every non-empty `OPENAI_API_KEY*`
    slot found, in order, so `client_for` can hand `OpenAIClient` the full rotation list.
    """
    load_env(env_path)
    anthropic_key = os.environ.get(_ANTHROPIC_KEY_ENV_VAR) or None
    openai_keys = [os.environ.get(name) for name in _OPENAI_KEY_ENV_VARS]
    return anthropic_key, [k for k in openai_keys if k]
