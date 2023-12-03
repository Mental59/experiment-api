from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from .mlflow_tracker import MLFlowTracker
from .neptune_tracker import NeptuneTracker
from .experiment_tracker import ExperimentTracker


def get_experiment_tracker(experiment_tracker_type: ExperimentTrackerEnum, project: str, run_name: str, **kwargs) -> ExperimentTracker:
    if experiment_tracker_type == ExperimentTrackerEnum.MLflow:
        return MLFlowTracker(project=project, run_name=run_name)
    elif experiment_tracker_type == ExperimentTrackerEnum.Neptune:
        return NeptuneTracker(project=project, run_name=run_name, api_token=kwargs['api_token'])
