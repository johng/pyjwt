"""
Type aliases for the ``cryptography`` key types that PyJWT accepts.

They live in their own module so that :mod:`jwt.algorithms` can import them
lazily (PEP 810): building the aliases imports ``cryptography``. Import them
from :mod:`jwt.algorithms`.
"""

import sys
from typing import Union

from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed448 import (
    Ed448PrivateKey,
    Ed448PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    # Python 3.9 and lower
    from typing_extensions import TypeAlias

# Type aliases for convenience in algorithms method signatures
AllowedRSAKeys: TypeAlias = Union[RSAPrivateKey, RSAPublicKey]
AllowedECKeys: TypeAlias = Union[EllipticCurvePrivateKey, EllipticCurvePublicKey]
AllowedOKPKeys: TypeAlias = Union[
    Ed25519PrivateKey, Ed25519PublicKey, Ed448PrivateKey, Ed448PublicKey
]
AllowedKeys: TypeAlias = Union[AllowedRSAKeys, AllowedECKeys, AllowedOKPKeys]
#: Type alias for allowed ``cryptography`` private keys (requires ``cryptography`` to be installed)
AllowedPrivateKeys: TypeAlias = Union[
    RSAPrivateKey, EllipticCurvePrivateKey, Ed25519PrivateKey, Ed448PrivateKey
]
#: Type alias for allowed ``cryptography`` public keys (requires ``cryptography`` to be installed)
AllowedPublicKeys: TypeAlias = Union[
    RSAPublicKey, EllipticCurvePublicKey, Ed25519PublicKey, Ed448PublicKey
]
