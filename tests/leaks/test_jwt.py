"""Leak suite, case 3: JWT tampering. Only a genuine, current token for this
issuer and audience yields a Principal, and the tenant comes only from the
single organization in it. No Keycloak or database needed.
"""

import json
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization

from ragmt.auth.tokens import InvalidTokenError, NoTenantError, TokenValidator
from ragmt.domain import Principal
from tests.auth_helpers import (
    SUB,
    TENANT_ID,
    FakeJwks,
    SigningKey,
    b64url,
    claims,
    forge,
    forge_hs256,
    make_validator,
)

pytestmark = pytest.mark.leaks


@pytest.fixture(scope="module")
def key() -> SigningKey:
    return SigningKey(kid="kc-rs256")


@pytest.fixture(scope="module")
def other_key() -> SigningKey:
    return SigningKey(kid="kc-rs256")  # same kid, different key pair


@pytest.fixture
def validator(key: SigningKey) -> TokenValidator:
    return make_validator(FakeJwks(key))


async def test_valid_token_gives_sub_and_tenant(key: SigningKey, validator: TokenValidator) -> None:
    principal = await validator.validate(key.sign(claims()))
    assert principal == Principal(sub=SUB, tenant_id=TENANT_ID)


async def test_groups_claim_is_ignored(key: SigningKey, validator: TokenValidator) -> None:
    principal = await validator.validate(key.sign(claims(groups=["/admins"])))
    assert not hasattr(principal, "groups")


async def test_single_string_audience_is_accepted(
    key: SigningKey, validator: TokenValidator
) -> None:
    await validator.validate(key.sign(claims(aud="ragmt-api")))


async def test_clock_skew_within_leeway_is_tolerated(
    key: SigningKey, validator: TokenValidator
) -> None:
    now = claims()["iat"]
    await validator.validate(key.sign(claims(exp=now - 5)))


# --- 401: the token proves nothing ---------------------------------------------


def _strip_signature(token: str) -> str:
    return token.rsplit(".", 1)[0] + "."


def _tamper_payload(token: str, **changes: Any) -> str:
    header, _, signature = token.split(".")
    return ".".join([header, b64url(json.dumps(claims(**changes)).encode()), signature])


def _public_pem(key: SigningKey) -> bytes:
    return key.private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


ATTACKS = {
    "garbage": lambda k, o: "not-a-jwt",
    "empty": lambda k, o: "",
    "signature stripped": lambda k, o: _strip_signature(k.sign(claims())),
    "alg none": lambda k, o: forge({"alg": "none", "typ": "JWT", "kid": k.kid}, claims()),
    "alg None capitalised": lambda k, o: forge({"alg": "None", "kid": k.kid}, claims()),
    "no alg": lambda k, o: forge({"typ": "JWT", "kid": k.kid}, claims(), b"x"),
    "HS256 with the public key PEM as secret": lambda k, o: forge_hs256(
        claims(), _public_pem(k), k.kid
    ),
    "HS256 with the JWK modulus as secret": lambda k, o: forge_hs256(
        claims(), k.jwk()["n"].encode(), k.kid
    ),
    "payload tampered (other tenant)": lambda k, o: _tamper_payload(
        k.sign(claims()),
        organization={"umbra": {"id": "17033638-c3cc-4d1e-b1fb-539941854a6e"}},
    ),
    "payload tampered (other sub)": lambda k, o: _tamper_payload(k.sign(claims()), sub="admin"),
    "signed by another key with the same kid": lambda k, o: o.sign(claims()),
    "unknown kid": lambda k, o: k.sign(claims(), kid="forged-kid"),
    "no kid": lambda k, o: forge({"alg": "RS256", "typ": "JWT"}, claims(), b"x"),
    "expired": lambda k, o: k.sign(claims(exp=claims()["iat"] - 60)),
    "not yet valid": lambda k, o: k.sign(claims(nbf=claims()["iat"] + 600)),
    "no exp": lambda k, o: k.sign(claims(drop=("exp",))),
    "wrong aud": lambda k, o: k.sign(claims(aud=["account", "other-api"])),
    "no aud": lambda k, o: k.sign(claims(drop=("aud",))),
    "wrong iss": lambda k, o: k.sign(claims(iss="http://keycloak.test/realms/other")),
    "iss with trailing slash": lambda k, o: k.sign(
        claims(iss="http://keycloak.test/realms/ragmt/")
    ),
    "no iss": lambda k, o: k.sign(claims(drop=("iss",))),
    "no sub": lambda k, o: k.sign(claims(drop=("sub",))),
    "empty sub": lambda k, o: k.sign(claims(sub="")),
    "blank sub": lambda k, o: k.sign(claims(sub="   ")),
    "non-string sub": lambda k, o: k.sign(claims(sub=42)),
}


@pytest.mark.parametrize("attack", ATTACKS)
async def test_untrusted_tokens_are_rejected(
    attack: str, key: SigningKey, other_key: SigningKey, validator: TokenValidator
) -> None:
    token = ATTACKS[attack](key, other_key)
    with pytest.raises(InvalidTokenError):
        await validator.validate(token)


async def test_reasons_never_contain_the_token(key: SigningKey, validator: TokenValidator) -> None:
    token = key.sign(claims(aud="other-api"))
    with pytest.raises(InvalidTokenError) as excinfo:
        await validator.validate(token)
    for part in token.split("."):
        assert part not in excinfo.value.reason


# --- 403: genuine token, but no single tenant -----------------------------------

ORGANIZATIONS = {
    "missing": None,
    "empty object": {},
    "two organizations": {
        "acme": {"id": str(TENANT_ID)},
        "umbra": {"id": "17033638-c3cc-4d1e-b1fb-539941854a6e"},
    },
    "list of aliases (mapper without id)": ["acme"],
    "single alias string": "acme",
    "no id": {"acme": {}},
    "null id": {"acme": {"id": None}},
    "numeric id": {"acme": {"id": 42}},
    "id is not a UUID": {"acme": {"id": "acme"}},
    "id with braces": {"acme": {"id": "{" + str(TENANT_ID) + "}"}},
    "id as urn": {"acme": {"id": TENANT_ID.urn}},
    "id as bare hex": {"acme": {"id": TENANT_ID.hex}},
    "id upper-cased": {"acme": {"id": str(TENANT_ID).upper()}},
    "organization is a string": {"acme": str(TENANT_ID)},
}


@pytest.mark.parametrize("case", ORGANIZATIONS)
async def test_tokens_without_exactly_one_organization_have_no_tenant(
    case: str, key: SigningKey, validator: TokenValidator
) -> None:
    value = ORGANIZATIONS[case]
    token_claims = claims(drop=("organization",)) if value is None else claims(organization=value)
    with pytest.raises(NoTenantError):
        await validator.validate(key.sign(token_claims))


async def test_tenant_id_elsewhere_in_the_token_is_ignored(
    key: SigningKey, validator: TokenValidator
) -> None:
    other = "17033638-c3cc-4d1e-b1fb-539941854a6e"
    principal = await validator.validate(key.sign(claims(tenant_id=other, org_id=other)))
    assert principal.tenant_id == TENANT_ID


# --- configuration --------------------------------------------------------------


@pytest.mark.parametrize("algorithms", [("none",), ("HS256",), ("RS256", "HS512"), ()])
def test_validator_refuses_unsafe_algorithms(key: SigningKey, algorithms: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="unsafe or empty"):
        make_validator(FakeJwks(key), algorithms=algorithms)


async def test_algorithm_outside_the_allow_list_is_rejected_before_any_fetch(
    key: SigningKey,
) -> None:
    fake = FakeJwks(key)
    validator = make_validator(fake, algorithms=("ES256",))
    with pytest.raises(InvalidTokenError):
        await validator.validate(key.sign(claims()))
    assert fake.calls == 0
