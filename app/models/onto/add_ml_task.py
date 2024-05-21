from pydantic import BaseModel


class AddMLTask(BaseModel):
    new_node_name: str
    parent_node_id: str | None = None
