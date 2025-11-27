from uuid import UUID

from advanced_alchemy.exceptions import IntegrityError
from fastapi import APIRouter, HTTPException, status

from src.auth.security.dependencies import CurrentAdminDep
from src.users.dependencies import UserServiceDep
from src.users.models import UserModel
from src.users.schemas import User, UserCreate, UserList

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/{user_id}",
    summary="Get user by ID",
    description="Retrieve a user by their unique identifier (UUID)",
    responses={
        status.HTTP_200_OK: {
            "description": "User found successfully",
        },
    },
)
async def get_user(user_id: str, service: UserServiceDep, admin: CurrentAdminDep) -> User:
    try:
        user_uuid = UUID(user_id)
    except ValueError as err:
        raise HTTPException(
            status_code=422,
            detail="Invalid UUID format",
            headers={"X-Error-Code": "INVALID_UUID"},
        ) from err

    user = await service.get(user_uuid)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
            headers={"X-Error-Code": "USER_NOT_FOUND"},
        )
    return user


@router.get(
    "",
    summary="List all users",
    description="Retrieve a list of all registered users",
    responses={
        status.HTTP_200_OK: {"description": "List of all users returned successfully"},
    },
)
async def get_all_users(service: UserServiceDep, admin: CurrentAdminDep) -> UserList:
    users = await service.list()
    return UserList(users=users)


@router.post(
    "",
    summary="Create a new user",
    status_code=status.HTTP_201_CREATED,
    description="""
    Register a new user in the system
    """,
    responses={
        status.HTTP_201_CREATED: {"description": "User was created successfully"},
    },
)
async def create_user(user: UserCreate, service: UserServiceDep) -> User:
    user_data = user.model_dump(exclude_unset=True, exclude_none=True)

    try:
        created_user = await service.create_user(UserModel(**user_data))
    except IntegrityError as err:
        raise HTTPException(
            status_code=400,
            detail="User with this username already exists",
            headers={"X-Error-Code": "USER_EXISTS"},
        ) from err

    return created_user
