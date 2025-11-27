from advanced_alchemy.extensions.fastapi import service

from src.auth.security.passwords import hash_password
from src.users.models import Role, UserModel
from src.users.repositories import UserRepository


class UserService(service.SQLAlchemyAsyncRepositoryService[UserModel, UserRepository]):
    """User Service"""

    repository_type = UserRepository

    async def create_user(self, user_obj: UserModel) -> UserModel:
        user_obj.password = hash_password(user_obj.password)
        return await self.create(user_obj, auto_commit=True)

    async def create_admin(self, username: str, password: str) -> UserModel:
        hashed_password = hash_password(password)
        return await self.create(
            UserModel(
                username=username,
                password=hashed_password,
                role=Role.ADMIN,
            )
        )
