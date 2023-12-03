from fastapi import APIRouter

from ...services.experiment.tracker_info import mlflow_tracker_info, neptune_tracker_info
from ...models.experiment.tracker_info import ExperimentTrackerInfo

router = APIRouter()


@router.get('/mlflow-tracker-info')
def get_mlflow_info() -> ExperimentTrackerInfo:
    return mlflow_tracker_info.get_experiment_tracker_info()


@router.get('/neptune-tracker-info')
def get_neptune_info(project: str, api_token: str) -> ExperimentTrackerInfo:
    return neptune_tracker_info.get_experiment_tracker_info(api_token=api_token, project_name=project)
