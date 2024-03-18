from sqlalchemy import Column, String, UUID
from uuid import uuid4

from .. import database


class UserDB(database.Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    login = Column(String, unique=True, index=True)
    password = Column(String)
