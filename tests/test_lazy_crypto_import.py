"""
``cryptography`` is only imported when an algorithm that needs it is used, so
that applications using only HMAC do not pay for it.
"""

import subprocess
import sys
import textwrap
from typing import Any, Callable

import pytest

from jwt import PyJWS, algorithms
from jwt.algorithms import HMACAlgorithm, requires_cryptography
from jwt.exceptions import InvalidKeyError

from .keys import load_ec_pub_key_p_521, load_rsa_pub_key
from .utils import crypto_required, no_crypto_required

if algorithms.has_crypto:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from jwt import _crypto_algorithms


def run_in_fresh_interpreter(code: str) -> None:
    """Run ``code`` in a new interpreter, which has not imported cryptography."""
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_hmac_does_not_import_cryptography() -> None:
    run_in_fresh_interpreter(
        """
        import sys

        import jwt
        from jwt.algorithms import HMACAlgorithm

        key = "a-secret-that-is-long-enough-for-hs256"
        token = jwt.encode({"sub": "someone"}, key, algorithm="HS256")
        assert jwt.decode(token, key, algorithms=["HS256"]) == {"sub": "someone"}
        jwk = jwt.PyJWK({"kty": "oct", "k": "c2VjcmV0"})
        assert jwk.algorithm_name == "HS256"
        HMACAlgorithm(HMACAlgorithm.SHA256).compute_hash_digest(b"data")
        jwt.PyJWS().get_algorithms()

        assert "cryptography" not in sys.modules, "cryptography was imported"
        """
    )


@crypto_required
def test_hmac_key_that_could_be_der_is_checked_with_cryptography() -> None:
    # DER keys start with 0x30 (ASCII "0"), so such secrets still get the
    # DER key check, which needs cryptography.
    run_in_fresh_interpreter(
        """
        import sys

        from jwt.algorithms import HMACAlgorithm

        HMACAlgorithm(HMACAlgorithm.SHA256).prepare_key(b"0 might be DER")

        assert "cryptography" in sys.modules
        """
    )


@crypto_required
@pytest.mark.parametrize(
    "load_public_key",
    [
        load_rsa_pub_key,
        load_ec_pub_key_p_521,
        lambda: Ed25519PrivateKey.generate().public_key(),
    ],
    ids=["RSA", "EC", "Ed25519"],
)
def test_hmac_prepare_key_rejects_der_public_keys(
    load_public_key: Callable[[], Any],
) -> None:
    der_key = load_public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    assert der_key.startswith(b"\x30")

    with pytest.raises(InvalidKeyError, match="asymmetric key"):
        HMACAlgorithm(HMACAlgorithm.SHA256).prepare_key(der_key)


@crypto_required
def test_crypto_names_are_available_from_jwt_algorithms() -> None:
    assert algorithms.has_crypto is True
    for name in (
        "RSAAlgorithm",
        "RSAPSSAlgorithm",
        "ECAlgorithm",
        "OKPAlgorithm",
        "AllowedPrivateKeys",
        "AllowedPublicKeys",
    ):
        assert getattr(algorithms, name) is getattr(_crypto_algorithms, name)


@no_crypto_required
def test_crypto_names_are_unavailable_without_cryptography() -> None:
    if sys.version_info >= (3, 11):
        from typing import Never
    else:
        from typing_extensions import Never

    assert algorithms.has_crypto is False
    assert algorithms.AllowedPrivateKeys is Never
    assert algorithms.AllowedPublicKeys is Never
    with pytest.raises(AttributeError):
        algorithms.RSAAlgorithm  # noqa: B018


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NotAnAlgorithm'"):
        algorithms.NotAnAlgorithm  # noqa: B018


@crypto_required
def test_crypto_algorithm_names_match_requires_cryptography() -> None:
    assert set(_crypto_algorithms.get_crypto_algorithms()) == requires_cryptography


@crypto_required
def test_pyjws_creates_crypto_algorithm_once() -> None:
    jws = PyJWS()

    algorithm = jws.get_algorithm_by_name("RS256")

    assert isinstance(algorithm, _crypto_algorithms.RSAAlgorithm)
    assert jws.get_algorithm_by_name("RS256") is algorithm


@crypto_required
def test_pyjws_disallowed_crypto_algorithm_is_not_supported() -> None:
    jws = PyJWS(algorithms=["HS256"])

    with pytest.raises(NotImplementedError, match="Algorithm not supported"):
        jws.get_algorithm_by_name("RS256")
