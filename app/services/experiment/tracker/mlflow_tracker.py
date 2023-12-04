import os
import json
import uuid

import pandas as pd
import mlflow
from matplotlib.figure import Figure
from neptune.types import File

from .experiment_tracker import ExperimentTracker
from ...dataset_processor.colorizer import colorize_html_table
from ...file_manager.temp_folder import TempFolder
from ....models.ml.experiment_run_result import ExperimentRunResult
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum


class MLFlowTracker(ExperimentTracker):
    def __init__(self, project: str, run_name: str):
        super().__init__(project=project, run_name=run_name)
        self.run = None
    
    def __enter__(self):
        mlflow.set_experiment(experiment_name=self.project)
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tbf):
        mlflow.end_run()
    
    def log_json(self, name: str, data: dict) -> None:
        mlflow.log_text(json.dumps(data), name + '.json')
    
    def log_txt(self, name: str, data: str) -> None:
        mlflow.log_text(data, name + '.txt')
    
    def log_dataset(self, name: str, dataset_path: str) -> None:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            content = file.read()
            mlflow.log_text(content, name + '.txt')
    
    def log_table(self, name: str, df: pd.DataFrame) -> None:
        mlflow.log_text(File.as_html(df).content.decode('utf-8'), name + '.html')

    def get_run_result(self) -> ExperimentRunResult:
        return ExperimentRunResult(
            experiment_tracker=ExperimentTrackerEnum.MLflow,
            project_id=self.run.info.experiment_id,
            run_id=self.run.info.run_id,
            run_name=self.run_name
        )
    
    def log_colorized_table(
        self,
        name: str,
        df: pd.DataFrame,
        matched_indices: list[tuple[int, int]],
        false_positive_indices: list[tuple[int, int]],
        false_negative_indices: list[tuple[int, int]]
    ) -> None:
        colorized_table = colorize_html_table(
            matched_indices,
            false_positive_indices,
            false_negative_indices,
            File.as_html(df).content
        )
        mlflow.log_text(colorized_table, name + '.html')
    
    def log_figure(self, name: str, figure: Figure) -> None:
        with TempFolder(str(uuid.uuid4())) as temp_folder:
            data_path = os.path.join(temp_folder.path, os.path.basename(name)) + '.png'
            figure.savefig(data_path)
            mlflow.log_artifact(data_path, os.path.split(name)[0])

    def log_params(self, name: str, param: dict[str]) -> None:
        params = {f'{name}/{key}': param[key] for key in param}
        mlflow.log_params(params)
    
    def log_model_pth(self, name: str, data: bytes) -> None:
        with TempFolder(str(uuid.uuid4())) as temp_folder:
            data_path = os.path.join(temp_folder.path, os.path.basename(name)) + '.pth'
            with open(data_path, 'wb') as file:
                file.write(data)
            mlflow.log_artifact(data_path, os.path.split(name)[0])
    
    def log_metric(self, name: str, param) -> None:
        mlflow.log_metric(name, param)

    def log_metrics(self, metrics: dict[str, int | float]) -> None:
        mlflow.log_metrics(metrics)

    def add_tags(self, tags: dict[str]) -> None:
        mlflow.set_tags(tags)
