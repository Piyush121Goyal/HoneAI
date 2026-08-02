"""Authentication routes.

Access token is returned in the JSON body (kept in memory on the client).
Refresh token is set as an httpOnly cookie and rotated on every refresh.
"""
import jwt
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import GoogleIn, LoginIn, SignupIn, TokenOut, UserOut
from ..security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "hone_refresh"


def _set_refresh_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=create_refresh_token(user_id),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400,
        domain=settings.cookie_domain,
        path="/auth",
    )


def _token_response(response: Response, user: User) -> TokenOut:
    _set_refresh_cookie(response, user.id)
    return TokenOut(
        access_token=create_access_token(user.id),
        user=UserOut.model_validate(user),
    )


@router.post("/signup", response_model=TokenOut, status_code=201)
async def signup(body: SignupIn, response: Response, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = User(
        email=body.email,
        name=body.name,
        hashed_password=hash_password(body.password),
        provider="password",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return _token_response(response, user)


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, response: Response, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == body.email))
    if not user or not user.hashed_password or not verify_password(
        body.password, user.hashed_password
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return _token_response(response, user)


@router.post("/refresh", response_model=TokenOut)
async def refresh(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No refresh token")
    try:
        payload = decode_token(token, expected_type="refresh")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")
    user = await db.get(User, payload.get("sub"))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return _token_response(response, user)  # rotates the cookie


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(REFRESH_COOKIE, path="/auth", domain=settings.cookie_domain)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.post("/google", response_model=TokenOut)
async def google(body: GoogleIn, response: Response, db: AsyncSession = Depends(get_db)):
    """Verify a Google ID token, then upsert the user and issue our own JWTs."""
    if not settings.google_client_id:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Google login not configured")
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": body.id_token},
        )
    if r.status_code != 200:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google token")
    info = r.json()
    if info.get("aud") != settings.google_client_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token audience mismatch")

    email = info["email"]
    user = await db.scalar(select(User).where(User.email == email))
    if not user:
        user = User(
            email=email,
            name=info.get("name") or email.split("@")[0],
            provider="google",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return _token_response(response, user)
