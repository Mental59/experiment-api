import os
import json
import uuid

import pandas as pd
import mlflow
from matplotlib.figure import Figure
from neptune.types import File

from .experiment_logger import ExperimentLogger
from ...dataset_processor.colorizer import colorize_html_table
from ....constants.artifacts import PATHS
from ...file_manager.temp_folder import TempFolder


class MLFlowLogger(ExperimentLogger):
    def __init__(self, project: str):
        super().__init__(project)
    
    def __enter__(self):
        mlflow.set_experiment(experiment_name=self.project)
        mlflow.start_run()
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
        with TempFolder(os.path.join(PATHS['TEMP_PATH'], str(uuid.uuid4()))) as temp_folder:
            data_path = os.path.join(temp_folder.path, os.path.basename(name)) + '.png'
            figure.savefig(data_path)
            mlflow.log_artifact(data_path, os.path.split(name)[0])
    
    def log_param(self, name: str, param) -> None:
        mlflow.log_param(name, param)
    
    def log_binary(self, name: str, data: bytes, extension: str) -> None:
        with TempFolder(os.path.join(PATHS['TEMP_PATH'], str(uuid.uuid4()))) as temp_folder:
            data_path = os.path.join(temp_folder.path, os.path.basename(name)) + f'.{extension}'
            with open(data_path, 'wb') as file:
                file.write(data)
            mlflow.log_artifact(data_path, os.path.split(name)[0])
    
    def append_param(self, name: str, param) -> None:
        mlflow.log_metric(name, param)

    def add_tags(self, tags: dict[str]) -> None:
        mlflow.set_tags(tags)
