from datetime import timedelta
from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException

from src.auth.security.schemas import TokenPayload
from src.auth.security.token import (
    create_access_token,
    create_refresh_token,
    create_token,
    get_data_from_payload,
    parse_token_payload,
    verify_refresh_token,
    verify_token,
)
from src.config import settings
from src.users.models import Role


@pytest.fixture
def token_payload():
    """Fixture for generating a sample TokenPayload with UUID and role."""
    return TokenPayload(id=uuid4(), role=Role.USER)


def test_create_token_returns_valid_jwt(token_payload):
    """Test that create_token returns a valid JWT with expected payload and exp."""
    data = {"id": str(token_payload.id), "role": token_payload.role.value}
    token = create_token(data, settings.SECRET_KEY, timedelta(minutes=5))

    decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    assert decoded["id"] == str(token_payload.id)
    assert decoded["role"] == token_payload.role.value
    assert "exp" in decoded


def test_get_data_from_payload(token_payload):
    """Test that get_data_from_payload returns correct dictionary from TokenPayload."""
    result = get_data_from_payload(token_payload)
    assert result["id"] == str(token_payload.id)
    assert result["role"] == token_payload.role.value


def test_create_access_token_returns_string(token_payload):
    """Test that create_access_token returns a string."""
    token = create_access_token(token_payload)
    assert isinstance(token, str)


def test_create_refresh_token_returns_string(token_payload):
    """Test that create_refresh_token returns a string."""
    token = create_refresh_token(token_payload)
    assert isinstance(token, str)


def test_parse_token_payload_valid(token_payload):
    """Test that parse_token_payload returns TokenPayload when input is valid."""
    payload = {"id": str(token_payload.id), "role": token_payload.role.value}
    result = parse_token_payload(payload, "error")
    assert result.id == token_payload.id
    assert result.role == token_payload.role


@pytest.mark.parametrize(
    "payload",
    [
        {"id": None, "role": "user"},
        {"id": "123", "role": None},
        {"id": "123", "role": "notarole"},
    ],
)
def test_parse_token_payload_invalid(payload):
    """Test that parse_token_payload raises HTTPException on invalid or missing fields."""
    with pytest.raises(HTTPException):
        parse_token_payload(payload, "invalid token")


def test_verify_token_valid(token_payload):
    """Test that verify_token returns valid TokenPayload when token is correct."""
    token = create_access_token(token_payload)
    result = verify_token(token)
    assert result.id == token_payload.id
    assert result.role == token_payload.role


def test_verify_refresh_token_valid(token_payload):
    """Test that verify_refresh_token returns valid TokenPayload when token is correct."""
    token = create_refresh_token(token_payload)
    result = verify_refresh_token(token)
    assert result.id == token_payload.id
    assert result.role == token_payload.role


def test_verify_token_expired(token_payload):
    """Test that verify_token raises HTTPException on expired token."""
    token = create_token(
        {"id": str(token_payload.id), "role": token_payload.role.value},
        settings.SECRET_KEY,
        timedelta(seconds=-1),  # already expired
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(token)
    assert "expired" in str(exc.value.detail).lower()


def test_verify_token_invalid():
    """Test that verify_token raises HTTPException on completely invalid token."""
    invalid_token = "this.is.not.valid"
    with pytest.raises(HTTPException) as exc:
        verify_token(invalid_token)
    assert "invalid" in str(exc.value.detail).lower()
