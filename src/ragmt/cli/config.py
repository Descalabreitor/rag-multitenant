"""Where the CLI connects and where it keeps its tokens.

The API URL and the issuer come from a flag, else the environment
(`RAGMT_API_URL`, `OIDC_ISSUER`, the same name the API reads), else the local
development defaults. Plain http is accepted only for loopback hosts, so a
bearer token never crosses a network unencrypted.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_ISSUER = "http://localhost:8080/realms/ragmt"
CLI_CLIENT_ID = "ragmt-cli"
# DEV ONLY (ADR 0005): password grant, disabled unless KC_DEV_PASSWORD_CLIENT=true at import.
DEV_CLIENT_ID = "ragmt-dev-password"
DEV_CREDENTIAL_ENV = "KC_DEMO_USER_PASSWORD"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ConfigError(ValueError):
    """A URL the CLI refuses to use."""


def check_url(name: str, value: str) -> str:
    """`value` without a trailing slash, if it is https, or http to a loopback host."""
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"{name} must be an http(s) URL")
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ConfigError(f"{name} must use https unless it points at this machine")
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ConfigError(f"{name} must not carry a query, fragment or credentials")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class CliConfig:
    api_url: str
    issuer: str
    client_id: str = CLI_CLIENT_ID

    @classmethod
    def resolve(
        cls,
        *,
        api_url: str | None,
        issuer: str | None,
        environ: Mapping[str, str] = os.environ,
    ) -> "CliConfig":
        """Flags first, then the environment, then the development defaults."""
        return cls(
            api_url=check_url(
                "API URL", api_url or environ.get("RAGMT_API_URL") or DEFAULT_API_URL
            ),
            issuer=check_url("issuer", issuer or environ.get("OIDC_ISSUER") or DEFAULT_ISSUER),
        )


def config_dir(environ: Mapping[str, str] = os.environ) -> Path:
    """The per-user config directory for ragctl (`RAGCTL_CONFIG_DIR` overrides it)."""
    override = environ.get("RAGCTL_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = environ.get("APPDATA")
        return Path(base) / "ragmt" if base else Path.home() / "AppData" / "Roaming" / "ragmt"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ragmt"
    xdg = environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "ragmt"
