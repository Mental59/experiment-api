from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.exceptions import create_exception_details
from app.db import models
from app.db import schemas
from app.db import queries
from app.db.dependecies import get_db
from app.services.auth.auth import Token, authenticate_user, create_access_token, get_current_active_user

router = APIRouter()


@router.post('/signup', response_model=schemas.User)
def signup(user: schemas.UserSignup, db: Session = Depends(get_db)):
    db_user = queries.get_user_by_login(db, user.login)
    if db_user:
        raise HTTPException(status_code=400, detail=create_exception_details(message="User with such login exists"))
    return queries.create_user(db, user)


@router.post('/signin')
def signin(signin_data: schemas.UserSignin, db: Session = Depends(get_db)) -> Token:
    user = authenticate_user(db, signin_data.login, signin_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=create_exception_details(message="Incorrect login or password"),
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user)
    return Token(access_token=access_token, token_type='bearer')


@router.get('/whoami', response_model=schemas.User)
def whoami(current_user: Annotated[models.UserDB, Depends(get_current_active_user)]):
    return current_user
