from pydantic import BaseModel


class GenerateOntologyInput(BaseModel):
    python_version: str
