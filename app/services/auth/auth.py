from datetime import timedelta, timezone, datetime
from typing import Annotated

from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.core.exceptions import create_exception_details
from app.core.settings import get_settings
from app.db import models
from app.db.dependecies import get_db
from app.db.queries.user import get_user_by_id, get_user_by_login
from app.services.auth.crypto import verify_password


settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
algorithm = 'HS256'


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: str


def authenticate_user(db: Session, login: str, password: str):
    user = get_user_by_login(db, login=login)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(user: models.UserDB):
    to_encode = dict(sub=str(user.id))
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=algorithm)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=create_exception_details(message="Некорректные учетные данные"),
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    user = get_user_by_id(db, id=token_data.user_id)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: Annotated[models.UserDB, Depends(get_current_user)]):
    return current_user
