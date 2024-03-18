from sqlalchemy.orm import Session

from app.db import models
from app.db import schemas
from app.services.auth.crypto import hash_password


def create_user(db: Session, user: schemas.UserCreate) -> models.UserDB:
    db_user = models.UserDB(login=user.login, password=hash_password(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_all_users(db: Session) -> list[models.UserDB]:
    return db.query(models.UserDB).all()


def get_user_by_login(db: Session, login: str):
    return db.query(models.UserDB).filter(models.UserDB.login == login).first()


def get_user_by_id(db: Session, id: str):
    return db.query(models.UserDB).filter(models.UserDB.id == id).first()
