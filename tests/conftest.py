"""Shared pytest setup."""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set.

    CI exports the variables itself, so existing values always win. Locally this
    lets `pytest` find DATABASE_URL and friends without exporting them by hand;
    otherwise the `db` tests would be skipped silently.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(_ENV_FILE)
