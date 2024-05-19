from .bilstm_params import BiLSTMHyperParams
from .train_params import TrainParams
from .ml_base_input import MLBaseInput


class MLTrainExperimentInput(MLBaseInput):
    model_params: BiLSTMHyperParams = BiLSTMHyperParams()
    train_params: TrainParams = TrainParams()
