from fastapi import APIRouter, Depends

from app.services.auth.auth import get_current_active_user

from ...services.experiment.tracker_info import mlflow_tracker_info, neptune_tracker_info
from ...models.experiment.tracker_info import ExperimentTrackerInfo

router = APIRouter(dependencies=[Depends(get_current_active_user)])


@router.get('/mlflow-tracker-info')
def get_mlflow_info() -> ExperimentTrackerInfo:
    return mlflow_tracker_info.get_experiment_tracker_info()


@router.get('/neptune-tracker-info')
def get_neptune_info(api_token: str) -> ExperimentTrackerInfo:
    return neptune_tracker_info.get_experiment_tracker_info(api_token=api_token)


@router.get('/neptune-check-api-token')
def check_neptune_token(api_token: str) -> bool:
    return neptune_tracker_info.check_token(api_token)
