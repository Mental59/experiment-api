from pydantic import BaseModel, validator

from ...services.file_manager import file_manager


class MLBaseInput(BaseModel):
    dataset: str = 'dataset_name'
    run_name: str = 'test_api_run'

    @validator('dataset')
    def dataset_must_exist(cls, dataset: str):
        if dataset not in file_manager.get_datasets():
            raise ValueError(f"Dataset {dataset} doesn't exist")
        return dataset
    
    class Config:
        use_enum_values = True
        protected_namespaces = ()
