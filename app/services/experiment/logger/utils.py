from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from .mlflow_logger import MLFlowLogger
from .neptune_logger import NeptuneLogger
from .experiment_logger import ExperimentLogger


def get_experiment_tracker(tracker: ExperimentTrackerEnum, project: str, run_name: str, **kwargs) -> ExperimentLogger:
    if tracker == ExperimentTrackerEnum.MLflow:
        return MLFlowLogger(project=project, run_name=run_name)
    elif tracker == ExperimentTrackerEnum.Neptune:
        return NeptuneLogger(project=project, run_name=run_name, api_token=kwargs['api_token'])
