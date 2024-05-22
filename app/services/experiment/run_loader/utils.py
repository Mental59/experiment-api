from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from .mlflow_run_loader import MLFlowRunLoader
from .neptune_run_loader import NeptuneRunLoader


def get_run_loader(experiment_tracker_type: str, project: str, run_id: str, **kwargs):
    if experiment_tracker_type == ExperimentTrackerEnum.MLflow.value:
        return MLFlowRunLoader(project=project, run_id=run_id)
    elif experiment_tracker_type == ExperimentTrackerEnum.Neptune.value:
        return NeptuneRunLoader(project=project, run_id=run_id, api_token=kwargs['api_token'])
