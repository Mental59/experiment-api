import neptune

from ....models.experiment.tracker_info import ExperimentTrackerInfo, ExperimentInfo, RunInfo, RunType
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum


def get_experiment_tracker_info(api_token: str, project_name: str) -> ExperimentTrackerInfo:
    logger_info = ExperimentTrackerInfo(tracker=ExperimentTrackerEnum.Neptune, projects=[])

    project = neptune.Project(project=project_name, api_token=api_token)
    runs_table = project.fetch_runs_table().to_pandas().sort_values(by="sys/creation_time", ascending=False)

    run_info = [RunInfo(run_id=run['sys/id'], run_name=run['sys/name'], run_type=get_run_type_from_tags(run['sys/tags'].split(','))) for _, run in runs_table.iterrows()]

    logger_info.projects.append(ExperimentInfo(project_id=project_name, project_name=project_name, runs=run_info))

    return logger_info


def get_run_type_from_tags(tags: list[str]) -> RunType:
    print(tags)
    for tag in tags:
        if tag == RunType.Train:
            return RunType.Train
        if tag == RunType.Test:
            return RunType.Test
    
    return RunType.Unknown
