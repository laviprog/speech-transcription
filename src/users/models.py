from enum import Enum

from advanced_alchemy.base import UUIDAuditBase
from sqlalchemy import Boolean, String
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import Mapped, mapped_column


class Role(str, Enum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"


class UserModel(UUIDAuditBase):
    """User model."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password: Mapped[str]
    role: Mapped[Role] = mapped_column(SQLAlchemyEnum(Role), default=Role.USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
