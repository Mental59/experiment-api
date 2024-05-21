from pydantic import BaseModel


class AddTransformerModel(BaseModel):
    parent_node_id: str
    node_name: str
    model_name_or_path: str
