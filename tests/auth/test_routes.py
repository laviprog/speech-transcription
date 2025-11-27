import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from src.auth.security.passwords import hash_password
from src.main import app
from src.users.dependencies import provide_user_service
from src.users.models import Role


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class MockUser:
    def __init__(self):
        self.id = uuid.uuid4()
        self.username = "user"
        self.password = hash_password("password")
        self.role = Role.USER


class MockUserService:
    async def get_one_or_none(self, username: str):
        if username == "user":
            return MockUser()
        return None


@pytest.fixture(autouse=True)
def override_user_service():
    app.dependency_overrides[provide_user_service] = lambda: MockUserService()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def credentials():
    return {"username": "user", "password": "password"}


@pytest.mark.parametrize(
    "credentials, expected_status",
    [
        ({"username": "user", "password": "incorrect"}, 401),
        ({"username": "non-existent-user", "password": "password"}, 401),
        ({"username": "non-existent-user", "password": "wrong"}, 401),
    ],
)
@pytest.mark.asyncio
async def test_login_failure_cases(credentials, expected_status, client):
    """Test login failure with various invalid credentials"""
    response = await client.post("/auth/login", json=credentials)
    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_login_token_structure(credentials, client):
    response = await client.post("/auth/login", json=credentials)
    data = response.json()

    assert response.status_code == 200
    assert "access_token" in data
    assert "refresh_token" in data
    assert "token_type" in data
    assert data["token_type"].lower() == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_json(client):
    response = await client.post("/auth/login", json={"wrong": "data"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_no_payload(client):
    response = await client.post("/auth/login")
    assert response.status_code == 422
