"""Shared pytest setup."""

import os
from pathlib import Path

from hypothesis import settings

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

# Hypothesis profiles. "ci" is the default (CI and a plain `pytest`); `make leaks`
# picks "nightly" through HYPOTHESIS_PROFILE. Neither has a deadline: examples
# that go through PostgreSQL take as long as the machine and the table sizes
# make them, and a deadline would only make them flaky. The example counts are
# sized for the property-based leak test (tests/leaks/test_properties.py), which
# checks every user against the database after every step. `print_blob` prints a
# failing example in a form `@reproduce_failure` can replay.
settings.register_profile(
    "ci", max_examples=200, stateful_step_count=5, deadline=None, print_blob=True
)
settings.register_profile("nightly", settings.get_profile("ci"), max_examples=2000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
