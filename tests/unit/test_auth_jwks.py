"""The JWKS cache: one fetch per TTL, rate-limited refetches on unknown kids."""

import asyncio

import pytest

from ragmt.auth.jwks import JwksCache, JwksUnavailableError, UnknownKeyError
from ragmt.auth.tokens import InvalidTokenError
from tests.auth_helpers import (
    JWKS_URL,
    TTL_SECONDS,
    FakeClock,
    FakeJwks,
    SigningKey,
    b64url,
    claims,
    make_validator,
)


@pytest.fixture(scope="module")
def key() -> SigningKey:
    return SigningKey(kid="kc-1")


@pytest.fixture(scope="module")
def rotated_key() -> SigningKey:
    return SigningKey(kid="kc-2")


async def test_jwks_is_fetched_once_for_many_validations(key: SigningKey) -> None:
    fake = FakeJwks(key)
    validator = make_validator(fake)
    for _ in range(50):
        await validator.validate(key.sign(claims()))
    assert fake.calls == 1


async def test_concurrent_requests_share_one_fetch(key: SigningKey) -> None:
    fake = FakeJwks(key)
    validator = make_validator(fake)
    await asyncio.gather(*(validator.validate(key.sign(claims())) for _ in range(20)))
    assert fake.calls == 1


async def test_jwks_is_refetched_after_the_ttl(key: SigningKey) -> None:
    fake, clock = FakeJwks(key), FakeClock()
    validator = make_validator(fake, clock)
    await validator.validate(key.sign(claims()))
    clock.advance(TTL_SECONDS - 1)
    await validator.validate(key.sign(claims()))
    assert fake.calls == 1
    clock.advance(1)
    await validator.validate(key.sign(claims()))
    assert fake.calls == 2


async def test_unknown_kid_refetch_is_rate_limited(key: SigningKey) -> None:
    fake, clock = FakeJwks(key), FakeClock()
    validator = make_validator(fake, clock)
    await validator.validate(key.sign(claims()))
    assert fake.calls == 1

    forged = key.sign(claims(), kid="forged")
    # Right after a fetch: no refetch at all.
    for _ in range(100):
        with pytest.raises(InvalidTokenError):
            await validator.validate(forged)
    assert fake.calls == 1

    # After 30 s, one refetch; the next forged kids within 30 s cost nothing.
    clock.advance(30)
    for _ in range(100):
        with pytest.raises(InvalidTokenError):
            await validator.validate(forged)
    assert fake.calls == 2

    clock.advance(29)
    with pytest.raises(InvalidTokenError):
        await validator.validate(forged)
    assert fake.calls == 2


async def test_key_rotation_is_picked_up_on_unknown_kid(
    key: SigningKey, rotated_key: SigningKey
) -> None:
    fake, clock = FakeJwks(key), FakeClock()
    validator = make_validator(fake, clock)
    await validator.validate(key.sign(claims()))

    fake.keys = [key, rotated_key]  # Keycloak rotates
    clock.advance(30)
    await validator.validate(rotated_key.sign(claims()))
    assert fake.calls == 2


async def test_jwks_outage_fails_closed_and_backs_off(key: SigningKey) -> None:
    fake, clock = FakeJwks(key), FakeClock()
    fake.down = True
    validator = make_validator(fake, clock)
    with pytest.raises(JwksUnavailableError):
        await validator.validate(key.sign(claims()))
    with pytest.raises(JwksUnavailableError):
        await validator.validate(key.sign(claims()))
    assert fake.calls == 1

    fake.down = False
    clock.advance(30)
    await validator.validate(key.sign(claims()))
    assert fake.calls == 2


async def test_expired_cache_is_not_used_when_the_refresh_fails(key: SigningKey) -> None:
    fake, clock = FakeJwks(key), FakeClock()
    validator = make_validator(fake, clock)
    await validator.validate(key.sign(claims()))
    fake.down = True
    clock.advance(TTL_SECONDS)
    with pytest.raises(JwksUnavailableError):
        await validator.validate(key.sign(claims()))


async def test_only_asymmetric_signing_keys_are_loaded(key: SigningKey) -> None:
    fake = FakeJwks(key)
    fake.extra_entries = [
        {"kty": "oct", "kid": "hmac", "k": b64url(b"shared-secret")},
        {**SigningKey(kid="enc").jwk(), "use": "enc", "alg": "RSA-OAEP"},
        {**SigningKey(kid="oaep").jwk(), "alg": "RSA-OAEP"},
        {"kty": "RSA"},  # no kid
        "not-an-object",
    ]
    cache = JwksCache(fake.client(), JWKS_URL, ttl_seconds=TTL_SECONDS, clock=FakeClock())
    assert (await cache.get_key("kc-1")).key_id == "kc-1"
    for kid in ("hmac", "enc", "oaep"):
        with pytest.raises(UnknownKeyError):
            await cache.get_key(kid)


async def test_jwks_without_usable_keys_is_unavailable() -> None:
    fake = FakeJwks()
    fake.extra_entries = [{"kty": "oct", "kid": "hmac", "k": b64url(b"shared-secret")}]
    cache = JwksCache(fake.client(), JWKS_URL, ttl_seconds=TTL_SECONDS, clock=FakeClock())
    with pytest.raises(JwksUnavailableError):
        await cache.get_key("hmac")
