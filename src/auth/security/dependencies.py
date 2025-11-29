from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.security.schemas import TokenPayload
from src.auth.security.token import verify_token
from src.users.models import Role

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenPayload:
    return verify_token(token)


async def get_current_admin(
    user_data: Annotated[TokenPayload, Depends(get_current_user)],
):
    if user_data.role != Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")
    return user_data


CurrentUserDep = Annotated[TokenPayload, Depends(get_current_user)]
CurrentAdminDep = Annotated[TokenPayload, Depends(get_current_admin)]
