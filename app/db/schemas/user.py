from uuid import UUID
from pydantic import BaseModel, field_validator


class UserBase(BaseModel):
    login: str


class UserCreate(UserBase):
    password: str


class UserSignin(UserBase):
    password: str


class UserSignup(UserBase):
    password: str


class User(UserBase):
    id: str

    @field_validator('id', mode='before')
    @classmethod
    def transform_id(cls, id: UUID):
        return str(id)

    class Config:
        from_attributes = True
