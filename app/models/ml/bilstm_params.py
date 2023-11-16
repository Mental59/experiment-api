from pydantic import BaseModel


class BiLSTMHyperParams(BaseModel):
    embedding_dim: int = 64
    hidden_dim: int = 64
