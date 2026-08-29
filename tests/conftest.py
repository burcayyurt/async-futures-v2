"""Test-session defaults that keep results independent of the developer's .env.

``HyperliquidSettings`` loads ``.env`` from the working directory, so on a
machine that runs the bot every ``from_env()`` in the suite silently inherits
the live configuration. That is how two fee assertions passed here for weeks
and failed the moment CI ran them on a checkout with no ``.env``: locally
``MAKER_ENTRY_ENABLED=true`` made the entry fee 1.5 bps, while the declared
default is ``False`` and charges 4.5.

A test whose result depends on a file that is not in the repository is not
testing what it claims to. Blanking ``env_file`` for the session makes every
test read the declared defaults plus whatever it sets explicitly, so the suite
answers the same way here, in CI, and on the bot's host.

Environment variables are cleared too: they outrank ``.env`` in pydantic's
precedence, so leaving them would reopen the same hole from the shell side.
"""

from __future__ import annotations

import pytest

from src.core.config import HyperliquidSettings


def _env_aliases() -> set[str]:
    aliases = set()
    for field in HyperliquidSettings.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, str):
            aliases.add(alias)
    return aliases


@pytest.fixture(autouse=True, scope="session")
def _settings_ignore_dotenv() -> None:
    HyperliquidSettings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def _settings_ignore_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for alias in _env_aliases():
        monkeypatch.delenv(alias, raising=False)
