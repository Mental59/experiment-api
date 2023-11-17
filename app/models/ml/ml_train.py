from pydantic import BaseModel, validator

from ...services.file_manager import file_manager
from .model_enum import ModelEnum
from .bilstm_params import BiLSTMHyperParams
from .train_params import TrainParams
from .experiment_tracker_enum import ExperimentTrackerEnum


class MLTrainExperimentInput(BaseModel):
    model: ModelEnum = ModelEnum.BiLSTM_CRF
    experiment_tracker: ExperimentTrackerEnum = ExperimentTrackerEnum.Neptune
    dataset: str = 'dataset_name'
    run_name: str = 'run_name'
    model_params: BiLSTMHyperParams = BiLSTMHyperParams()
    train_params: TrainParams = TrainParams()

    @validator('dataset')
    def dataset_must_exist(cls, dataset: str):
        if dataset not in file_manager.get_datasets():
            raise ValueError(f"Dataset {dataset} doesn't exist")
        return dataset
        
    class Config:
        use_enum_values = True
        protected_namespaces = ()
