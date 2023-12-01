from pydantic import BaseModel, field_validator
from fastapi import HTTPException

from ...services.dataset_processor.loader import get_datasets


class MLBaseInput(BaseModel):
    dataset: str = 'dataset_name'
    run_name: str = 'test_api_run'

    @field_validator('dataset')
    @classmethod
    def dataset_must_exist(cls, dataset: str):
        if dataset not in get_datasets():
            raise HTTPException(status_code=400, detail=dict(message=f"Dataset {dataset} doesn't exist"))
        return dataset
    
    class Config:
        use_enum_values = True
        protected_namespaces = ()
