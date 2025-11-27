from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import jwt
from fastapi import HTTPException, status

from src.auth.security.schemas import TokenPayload
from src.config import settings
from src.users.models import Role


def create_token(data: dict[str, Any], secret: str, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm=settings.JWT_ALGORITHM)


def get_data_from_payload(payload: TokenPayload) -> dict[str, Any]:
    data = payload.model_dump()
    data["role"] = payload.role.value
    data["id"] = str(payload.id)
    return data


def create_access_token(token_payload: TokenPayload) -> str:
    data = get_data_from_payload(token_payload)
    return create_token(
        data=data,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        secret=settings.SECRET_KEY,
    )


def create_refresh_token(token_payload: TokenPayload) -> str:
    data = get_data_from_payload(token_payload)
    return create_token(
        data=data,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        secret=settings.SECRET_REFRESH_KEY,
    )


def create_tokens(token_payload: TokenPayload) -> tuple[str, str]:
    access_token = create_access_token(token_payload)
    refresh_token = create_refresh_token(token_payload)
    return access_token, refresh_token


def parse_token_payload(payload: dict[str, Any], error_detail: str) -> TokenPayload:
    user_id: str | None = payload.get("id")
    role_value = payload.get("role")

    if not user_id or not role_value or role_value not in Role._value2member_map_:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error_detail)

    return TokenPayload(id=UUID(user_id), role=Role(role_value))


def _verify_jwt_token(
    token: str,
    secret: str,
    expired_error_msg: str,
    invalid_error_msg: str,
) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return parse_token_payload(payload, invalid_error_msg)
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=expired_error_msg
        ) from err
    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=invalid_error_msg
        ) from err


def verify_refresh_token(token: str) -> TokenPayload:
    return _verify_jwt_token(
        token,
        settings.SECRET_REFRESH_KEY,
        "Refresh token expired",
        "Invalid refresh token",
    )


def verify_token(token: str) -> TokenPayload:
    return _verify_jwt_token(
        token,
        settings.SECRET_KEY,
        "Token expired",
        "Invalid token",
    )
