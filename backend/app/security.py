"""Auth primitives: argon2 password hashing + JWT access/refresh tokens."""
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from .config import settings

_ph = PasswordHasher()


# ----------------------------- passwords ----------------------------- #
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def needs_rehash(hashed: str) -> bool:
    try:
        return _ph.check_needs_rehash(hashed)
    except Exception:
        return False


# ------------------------------- tokens ------------------------------ #
def _encode(payload: dict, expires: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    to_encode = {
        **payload,
        "type": token_type,
        "iat": now,
        "exp": now + expires,
    }
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str) -> str:
    return _encode(
        {"sub": user_id},
        timedelta(minutes=settings.access_token_expire_minutes),
        "access",
    )


def create_refresh_token(user_id: str) -> str:
    return _encode(
        {"sub": user_id},
        timedelta(days=settings.refresh_token_expire_days),
        "refresh",
    )


def decode_token(token: str, expected_type: str | None = None) -> dict:
    """Decode and validate a JWT. Raises jwt exceptions on failure."""
    payload = jwt.decode(
        token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
    )
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("wrong token type")
    return payload
