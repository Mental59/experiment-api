from typing import Optional

from pydantic import BaseModel, validator

from ...services.file_manager import file_manager
from ...services.experiment import name_generator
from .model_enum import ModelEnum
from .bilstm_params import BiLSTMHyperParams
from .train_params import TrainParams


class MLTrainExperimentInput(BaseModel):
    model: ModelEnum = ModelEnum.BiLSTM_CRF
    dataset: str = 'dataset_name'
    run_name: str = 'run_name'
    model_params: BiLSTMHyperParams = BiLSTMHyperParams()
    train_params: TrainParams = TrainParams()

    @validator('dataset')
    def dataset_must_exist(cls, dataset: str):
        if dataset not in file_manager.get_datasets():
            raise ValueError(f"Dataset {dataset} doesn't exist")
        return dataset

    @validator('run_name')
    def generate_run_name(cls, run_name: Optional[str]):
        if run_name is None:
            return name_generator.get_experiment_id()
        
    class Config:
        use_enum_values = True
        protected_namespaces = ()
