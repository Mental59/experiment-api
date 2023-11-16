from typing import Optional

from pydantic import BaseModel

from .model_enum import ModelEnum

class MLTrain(BaseModel):
    model: ModelEnum = ModelEnum.BiLSTM_CRF
    run_name: Optional[str] = None

    class Config:
        use_enum_values = True
