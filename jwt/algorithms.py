from __future__ import annotations

import functools
import hashlib
import hmac
import importlib.util
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    overload,
)

from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
    base64url_decode,
    base64url_encode,
    force_bytes,
    is_pem_format,
    is_ssh_key,
)


if TYPE_CHECKING or bool(os.getenv("SPHINX_BUILD", "")):
    from cryptography.hazmat.primitives.asymmetric.types import (
        PrivateKeyTypes,
        PublicKeyTypes,
    )

if TYPE_CHECKING:
    has_crypto: bool

# Names that jwt.algorithms provides from jwt._crypto_algorithms. That module
# imports cryptography, so it is only imported when one of these names is
# first used (see __getattr__ below), not when jwt is imported. Type checkers
# and the API docs see them through the block at the end of this module.
_CRYPTO_NAMES = frozenset(
    {
        "AllowedECKeys",
        "AllowedKeys",
        "AllowedOKPKeys",
        "AllowedPrivateKeys",
        "AllowedPublicKeys",
        "AllowedRSAKeys",
        "ECAlgorithm",
        "OKPAlgorithm",
        "RSAAlgorithm",
        "RSAPSSAlgorithm",
    }
)


@functools.lru_cache(maxsize=None)
def _has_crypto() -> bool:
    """
    Import the ``cryptography``-backed algorithms, and return whether that
    worked (it does not if ``cryptography`` is not installed).

    This is the lazy equivalent of the module-level ``has_crypto`` flag.
    """
    try:
        from . import _crypto_algorithms  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


@functools.lru_cache(maxsize=None)
def _cryptography_installed() -> bool:
    """
    Whether ``cryptography`` is installed, checked without importing it.
    """
    return importlib.util.find_spec("cryptography") is not None


def __getattr__(name: str) -> Any:
    # PEP 562: resolve has_crypto and the cryptography-backed names on first
    # access, so that importing this module does not import cryptography.
    if name == "has_crypto":
        return _has_crypto()
    if name in _CRYPTO_NAMES:
        if _has_crypto():
            from . import _crypto_algorithms

            return getattr(_crypto_algorithms, name)
        if name.startswith("Allowed"):
            # Keep the type aliases importable without cryptography.
            if sys.version_info >= (3, 11):
                from typing import Never
            else:
                from typing_extensions import Never
            return Never
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


requires_cryptography = {
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES256K",
    "ES384",
    "ES521",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
    "EdDSA",
}


def _get_non_crypto_algorithms() -> dict[str, Algorithm]:
    return {
        "none": NoneAlgorithm(),
        "HS256": HMACAlgorithm(HMACAlgorithm.SHA256),
        "HS384": HMACAlgorithm(HMACAlgorithm.SHA384),
        "HS512": HMACAlgorithm(HMACAlgorithm.SHA512),
    }


def get_default_algorithms() -> dict[str, Algorithm]:
    """
    Returns the algorithms that are implemented by the library.
    """
    default_algorithms = _get_non_crypto_algorithms()

    if _has_crypto():
        from ._crypto_algorithms import get_crypto_algorithms

        default_algorithms.update(get_crypto_algorithms())

    return default_algorithms


def _get_default_algorithm(alg_name: str) -> Algorithm:
    """
    Returns a new instance of the default algorithm ``alg_name``, importing
    ``cryptography`` only if that algorithm needs it.

    :raises KeyError: if there is no such algorithm, or it needs
        ``cryptography`` and that is not installed.
    """
    if alg_name in requires_cryptography and _has_crypto():
        from ._crypto_algorithms import get_crypto_algorithms

        return get_crypto_algorithms()[alg_name]
    return _get_non_crypto_algorithms()[alg_name]


class Algorithm(ABC):
    """
    The interface for an algorithm used to sign and verify tokens.
    """

    # pyjwt-964: Validate to ensure the key passed in was decoded to the correct cryptography key family
    _crypto_key_types: tuple[type[AllowedKeys], ...] | None = None

    def compute_hash_digest(self, bytestr: bytes) -> bytes:
        """
        Compute a hash digest using the specified algorithm's hash algorithm.

        If there is no hash algorithm, raises a NotImplementedError.
        """
        # lookup self.hash_alg if defined in a way that mypy can understand
        hash_alg = getattr(self, "hash_alg", None)
        if hash_alg is None:
            raise NotImplementedError

        # If hash_alg is a cryptography hash algorithm, the module defining
        # its base class has already been imported, so look it up rather than
        # importing cryptography here.
        hashes = sys.modules.get("cryptography.hazmat.primitives.hashes")
        if (
            hashes is not None
            and isinstance(hash_alg, type)
            and issubclass(hash_alg, hashes.HashAlgorithm)
        ):
            from cryptography.hazmat.backends import default_backend

            digest = hashes.Hash(hash_alg(), backend=default_backend())
            digest.update(bytestr)
            return bytes(digest.finalize())
        else:
            return bytes(hash_alg(bytestr).digest())

    def check_crypto_key_type(self, key: PublicKeyTypes | PrivateKeyTypes) -> None:
        """Check that the key belongs to the right cryptographic family.

        Note that this method only works when ``cryptography`` is installed.

        :param key: Potentially a cryptography key
        :type key: :py:data:`PublicKeyTypes <cryptography.hazmat.primitives.asymmetric.types.PublicKeyTypes>` | :py:data:`PrivateKeyTypes <cryptography.hazmat.primitives.asymmetric.types.PrivateKeyTypes>`
        :raises ValueError: if ``cryptography`` is not installed, or this method is called by a non-cryptography algorithm
        :raises InvalidKeyError: if the key doesn't match the expected key classes
        """
        # _crypto_key_types is only set by cryptography-backed algorithms.
        if self._crypto_key_types is None:
            raise ValueError(
                "This method requires the cryptography library, and should only be used by cryptography-based algorithms."
            )

        if not isinstance(key, self._crypto_key_types):
            valid_classes = (cls.__name__ for cls in self._crypto_key_types)
            actual_class = key.__class__.__name__
            self_class = self.__class__.__name__
            raise InvalidKeyError(
                f"Expected one of {valid_classes}, got: {actual_class}. Invalid Key type for {self_class}"
            )

    @abstractmethod
    def prepare_key(self, key: Any) -> Any:
        """
        Performs necessary validation and conversions on the key and returns
        the key value in the proper format for sign() and verify().
        """

    @abstractmethod
    def sign(self, msg: bytes, key: Any) -> bytes:
        """
        Returns a digital signature for the specified message
        using the specified key value.
        """

    @abstractmethod
    def verify(self, msg: bytes, key: Any, sig: bytes) -> bool:
        """
        Verifies that the specified digital signature is valid
        for the specified message and key values.
        """

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(key_obj: Any, as_dict: Literal[True]) -> JWKDict: ...  # pragma: no cover

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(
        key_obj: Any, as_dict: Literal[False] = False
    ) -> str: ...  # pragma: no cover

    @staticmethod
    @abstractmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> JWKDict | str:
        """
        Serializes a given key into a JWK
        """

    @staticmethod
    @abstractmethod
    def from_jwk(jwk: str | JWKDict) -> Any:
        """
        Deserializes a given key from JWK back into a key object
        """

    def check_key_length(self, key: Any) -> str | None:
        """
        Return a warning message if the key is below the minimum
        recommended length for this algorithm, or None if adequate.
        """
        return None


class NoneAlgorithm(Algorithm):
    """
    Placeholder for use when no signing or verification
    operations are required.
    """

    def prepare_key(self, key: str | None) -> None:
        if key == "":
            key = None

        if key is not None:
            raise InvalidKeyError('When alg = "none", key value must be None.')

        return key

    def sign(self, msg: bytes, key: None) -> bytes:
        return b""

    def verify(self, msg: bytes, key: None, sig: bytes) -> bool:
        return False

    @staticmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> NoReturn:
        raise NotImplementedError()

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> NoReturn:
        raise NotImplementedError()


class HMACAlgorithm(Algorithm):
    """
    Performs signing and verification operations using HMAC
    and the specified hash function.
    """

    SHA256: ClassVar[HashlibHash] = hashlib.sha256
    SHA384: ClassVar[HashlibHash] = hashlib.sha384
    SHA512: ClassVar[HashlibHash] = hashlib.sha512

    def __init__(self, hash_alg: HashlibHash) -> None:
        self.hash_alg = hash_alg

    @staticmethod
    def _is_der_key(key_bytes: bytes) -> bool:
        # DER public keys (SubjectPublicKeyInfo or PKCS#1) and X.509
        # certificates are all ASN.1 SEQUENCEs, so their first byte is always
        # the identifier octet 0x30 (X.690, 8.1.2). Checking it first means
        # most HMAC secrets never need cryptography to be imported.
        if not key_bytes.startswith(b"\x30") or not _has_crypto():
            return False

        from ._crypto_algorithms import is_der_key

        return is_der_key(key_bytes)

    def prepare_key(self, key: str | bytes) -> bytes:
        key_bytes = force_bytes(key)

        if len(key_bytes) == 0:
            raise InvalidKeyError("HMAC key must not be empty.")

        if (
            is_pem_format(key_bytes)
            or is_ssh_key(key_bytes)
            or self._is_der_key(key_bytes)
        ):
            raise InvalidKeyError(
                "The specified key is an asymmetric key or x509 certificate and"
                " should not be used as an HMAC secret."
            )

        # Defense against algorithm-confusion attacks: an attacker with
        # control over the token header can force this code path by setting
        # alg=HS*, and HMACAlgorithm is the only algorithm that accepts
        # arbitrary bytes as a valid secret. Other algorithms reject
        # non-key-shaped input naturally. Even a symmetric (kty=oct) JWK
        # should be loaded via PyJWK / from_jwk rather than fed as raw JSON
        # bytes (whose contents are not the secret material).
        try:
            jwk_obj = json.loads(key_bytes, parse_int=lambda _: 0)
        except RecursionError:
            try:
                decoded_key = key_bytes.decode(
                    json.detect_encoding(key_bytes), errors="surrogatepass"
                )
            except UnicodeError:
                decoded_key = ""
            stripped_key = decoded_key.lstrip("\ufeff \t\r\n")
            has_jwk_member = False
            index = 0
            while index < len(decoded_key):
                if decoded_key[index] != '"':
                    index += 1
                    continue
                end = index + 1
                while end < len(decoded_key):
                    if decoded_key[end] == "\\":
                        end += 2
                    elif decoded_key[end] == '"':
                        break
                    else:
                        end += 1
                if end >= len(decoded_key):
                    break
                next_index = end + 1
                while (
                    next_index < len(decoded_key)
                    and decoded_key[next_index] in " \t\r\n"
                ):
                    next_index += 1
                if next_index < len(decoded_key) and decoded_key[next_index] == ":":
                    try:
                        has_jwk_member = (
                            json.loads(decoded_key[index : end + 1]) == "kty"
                        )
                    except ValueError:
                        pass
                    if has_jwk_member:
                        break
                index = end + 1
            if stripped_key.startswith("{") or (
                stripped_key.startswith("[") and has_jwk_member
            ):
                raise InvalidKeyError(
                    "The specified key looks like a JWK and should not be "
                    "used directly as an HMAC secret. Load it via "
                    "PyJWK / HMACAlgorithm.from_jwk first."
                ) from None
            jwk_obj = None
        except ValueError:
            jwk_obj = None
        contains_jwk_member = False
        objects_to_check = [jwk_obj]
        while objects_to_check:
            obj = objects_to_check.pop()
            if isinstance(obj, dict):
                if "kty" in obj:
                    contains_jwk_member = True
                    break
                objects_to_check.extend(obj.values())
            elif isinstance(obj, list):
                objects_to_check.extend(obj)
        if contains_jwk_member:
            raise InvalidKeyError(
                "The specified key looks like a JWK and should not be "
                "used directly as an HMAC secret. Load it via "
                "PyJWK / HMACAlgorithm.from_jwk first."
            )

        return key_bytes

    @overload
    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: Literal[True]) -> JWKDict: ...

    @overload
    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: Literal[False] = False) -> str: ...

    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: bool = False) -> JWKDict | str:
        jwk = {
            "k": base64url_encode(force_bytes(key_obj)).decode(),
            "kty": "oct",
        }

        if as_dict:
            return jwk
        else:
            return json.dumps(jwk)

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> bytes:
        try:
            if isinstance(jwk, str):
                obj: JWKDict = json.loads(jwk)
            elif isinstance(jwk, dict):
                obj = jwk
            else:
                raise ValueError
        except ValueError:
            raise InvalidKeyError("Key is not valid JSON") from None

        if obj.get("kty") != "oct":
            raise InvalidKeyError("Not an HMAC key")

        key_bytes = base64url_decode(obj["k"])
        if len(key_bytes) == 0:
            raise InvalidKeyError("HMAC key must not be empty.")
        return key_bytes

    def check_key_length(self, key: bytes) -> str | None:
        min_length = self.hash_alg().digest_size
        if len(key) < min_length:
            return (
                f"The HMAC key is {len(key)} bytes long, which is below "
                f"the minimum recommended length of {min_length} bytes for "
                f"{self.hash_alg().name.upper()}. "
                f"See RFC 7518 Section 3.2."
            )
        return None

    def sign(self, msg: bytes, key: bytes) -> bytes:
        return hmac.new(key, msg, self.hash_alg).digest()

    def verify(self, msg: bytes, key: bytes, sig: bytes) -> bool:
        return hmac.compare_digest(sig, self.sign(msg, key))


if TYPE_CHECKING or bool(os.getenv("SPHINX_BUILD", "")):
    # Bind the names that __getattr__ otherwise provides on first use. This is
    # at the end of the module because jwt._crypto_algorithms imports
    # Algorithm from here.
    from . import _crypto_algorithms

    AllowedRSAKeys = _crypto_algorithms.AllowedRSAKeys
    AllowedECKeys = _crypto_algorithms.AllowedECKeys
    AllowedOKPKeys = _crypto_algorithms.AllowedOKPKeys
    AllowedKeys = _crypto_algorithms.AllowedKeys
    #: Type alias for allowed ``cryptography`` private keys (requires ``cryptography`` to be installed)
    AllowedPrivateKeys = _crypto_algorithms.AllowedPrivateKeys
    #: Type alias for allowed ``cryptography`` public keys (requires ``cryptography`` to be installed)
    AllowedPublicKeys = _crypto_algorithms.AllowedPublicKeys
    RSAAlgorithm = _crypto_algorithms.RSAAlgorithm
    ECAlgorithm = _crypto_algorithms.ECAlgorithm
    RSAPSSAlgorithm = _crypto_algorithms.RSAPSSAlgorithm
    OKPAlgorithm = _crypto_algorithms.OKPAlgorithm
