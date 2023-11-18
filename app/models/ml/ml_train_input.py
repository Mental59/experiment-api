from .model_enum import ModelEnum
from .bilstm_params import BiLSTMHyperParams
from .train_params import TrainParams
from .experiment_tracker_enum import ExperimentTrackerEnum
from .ml_base_input import MLBaseInput


class MLTrainExperimentInput(MLBaseInput):
    model: ModelEnum = ModelEnum.BiLSTM_CRF
    experiment_tracker: ExperimentTrackerEnum = ExperimentTrackerEnum.Neptune
    model_params: BiLSTMHyperParams = BiLSTMHyperParams()
    train_params: TrainParams = TrainParams()
