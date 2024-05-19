from pydantic import BaseModel, field_validator
from fastapi import HTTPException

from app.models.ml.experiment_tracker_enum import ExperimentTrackerEnum

from ...services.dataset_processor.loader import get_datasets
from ...core.exceptions import create_exception_details


class MLBaseInput(BaseModel):
    dataset: str = 'dataset_name'
    run_name: str = 'test_api_run'
    experiment_tracker: ExperimentTrackerEnum = ExperimentTrackerEnum.MLflow
    base_experiment_id: str | None = None

    @field_validator('dataset')
    @classmethod
    def dataset_must_exist(cls, dataset: str):
        if dataset not in get_datasets():
            raise HTTPException(status_code=400, detail=create_exception_details(f"Dataset {dataset} doesn't exist"))
        return dataset
    
    class Config:
        use_enum_values = True
        protected_namespaces = ()
