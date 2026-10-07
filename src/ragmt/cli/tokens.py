"""The token file: what `ragctl login` keeps between runs, readable by the user only.

The file is written to a temporary file created with mode 0600 (in a 0700
directory) and then moved over the old one, so it is never readable by others,
not even for a moment, and a crash never leaves half a file. On Windows the
mode bits don't apply; the file lives under the user's profile (%APPDATA%),
which other users can't read by default.

A `TokenSet` never prints its tokens: its repr hides them.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILE_NAME = "tokens.json"
# Refresh this many seconds before the access token expires, so it doesn't
# expire between the check and the request.
EXPIRY_MARGIN_SECONDS = 30


class TokenFileError(ValueError):
    """The token file exists but can't be used."""


@dataclass(frozen=True, slots=True)
class TokenSet:
    """Tokens for one issuer and client. Times are Unix seconds."""

    issuer: str
    client_id: str
    access_token: str = field(repr=False)
    expires_at: float
    refresh_token: str | None = field(default=None, repr=False)
    # None: the refresh token's lifetime is unknown (or it doesn't expire).
    refresh_expires_at: float | None = None

    @classmethod
    def from_response(
        cls, body: Any, *, issuer: str, client_id: str, now: float | None = None
    ) -> "TokenSet":
        """From a token endpoint's JSON body (RFC 6749 section 5.1)."""
        if not isinstance(body, dict):
            raise TokenFileError("token response is not a JSON object")
        access_token = body.get("access_token")
        expires_in = body.get("expires_in")
        refresh_token = body.get("refresh_token")
        refresh_expires_in = body.get("refresh_expires_in")
        if not isinstance(access_token, str) or not access_token:
            raise TokenFileError("token response has no access_token")
        if not isinstance(expires_in, int | float) or isinstance(expires_in, bool):
            raise TokenFileError("token response has no expires_in")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise TokenFileError("token response has an invalid refresh_token")
        now = time.time() if now is None else now
        # Keycloak sends refresh_expires_in = 0 for tokens without an idle limit.
        refresh_expires_at = (
            now + refresh_expires_in
            if isinstance(refresh_expires_in, int | float) and refresh_expires_in > 0
            else None
        )
        return cls(
            issuer=issuer,
            client_id=client_id,
            access_token=access_token,
            expires_at=now + expires_in,
            refresh_token=refresh_token or None,
            refresh_expires_at=refresh_expires_at,
        )

    def access_valid(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now < self.expires_at - EXPIRY_MARGIN_SECONDS

    def can_refresh(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self.refresh_token is None:
            return False
        return self.refresh_expires_at is None or now < self.refresh_expires_at

    def to_json(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "client_id": self.client_id,
            "access_token": self.access_token,
            "expires_at": self.expires_at,
            "refresh_token": self.refresh_token,
            "refresh_expires_at": self.refresh_expires_at,
        }

    @classmethod
    def from_json(cls, data: Any) -> "TokenSet":
        try:
            return cls(
                issuer=str(data["issuer"]),
                client_id=str(data["client_id"]),
                access_token=str(data["access_token"]),
                expires_at=float(data["expires_at"]),
                refresh_token=(
                    None if data.get("refresh_token") is None else str(data["refresh_token"])
                ),
                refresh_expires_at=(
                    None
                    if data.get("refresh_expires_at") is None
                    else float(data["refresh_expires_at"])
                ),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise TokenFileError("token file is malformed") from exc


class TokenStore:
    """Reads and writes the token file in `directory`."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / FILE_NAME

    def load(self) -> TokenSet | None:
        """The stored tokens, or None if there is no file."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return TokenSet.from_json(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise TokenFileError("token file is malformed") from exc

    def save(self, tokens: TokenSet) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = directory / f".{FILE_NAME}.{os.getpid()}.tmp"
        # O_EXCL: never write through a file (or symlink) someone else put there.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(tokens.to_json(), fh)
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        # The umask can only remove bits, but make the mode explicit anyway.
        os.chmod(self.path, 0o600)

    def clear(self) -> bool:
        """Delete the token file. True if there was one."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True
