from fastapi import HTTPException

from app.core.exceptions import create_exception_details
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from .mlflow_tracker import MLFlowTracker
from .neptune_tracker import NeptuneTracker
from .experiment_tracker import ExperimentTracker


def get_experiment_tracker(experiment_tracker_type: str, project: str, run_name: str, **kwargs) -> ExperimentTracker:
    if experiment_tracker_type == ExperimentTrackerEnum.MLflow.value:
        return MLFlowTracker(project=project, run_name=run_name)
    elif experiment_tracker_type == ExperimentTrackerEnum.Neptune.value:
        api_token = kwargs.get('api_token')
        if not api_token:
            raise HTTPException(status_code=400, detail=create_exception_details(message='Не предоставлен API токен для Neptune'))
        return NeptuneTracker(project=project, run_name=run_name, api_token=api_token)

    raise HTTPException(status_code=400, detail=create_exception_details(message='Нeподдерживаемый тип инструмента отслеживания экспериментом'))
