from fastapi import APIRouter

from ...services.experiment.logger_info import mlflow_logger_info, neptune_logger_info
from ...models.experiment.logger_info import LoggerInfo

router = APIRouter()


@router.get('/mlflow-info')
def get_mlflow_info() -> LoggerInfo:
    return mlflow_logger_info.get_info()


@router.get('/neptune-info')
def get_neptune_info(project: str, api_token: str) -> LoggerInfo:
    return neptune_logger_info.get_info(api_token=api_token, project_name=project)
