import mlflow
from mlflow.entities import ViewType

from ....models.experiment.logger_info import LoggerInfo, ExperimentInfo, RunInfo, RunType
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum


def get_info() -> LoggerInfo:
    logger_info = LoggerInfo(tracker=ExperimentTrackerEnum.MLflow, projects=[])

    client = mlflow.tracking.MlflowClient()
    experiements = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    runs = client.search_runs(experiment_ids=[exp.experiment_id for exp in experiements], run_view_type=ViewType.ACTIVE_ONLY)

    for exp in experiements:
        exp_run_info = [
            RunInfo(
                run_id=run.info.run_id,
                run_name=run.info.run_name,
                run_type=run.data.tags.get('mode', RunType.Unknown)
            ) for run in runs if run.info.experiment_id == exp.experiment_id
        ]
        logger_info.projects.append(ExperimentInfo(project_id=exp.experiment_id, project_name=exp.name, runs=exp_run_info))

    return logger_info
