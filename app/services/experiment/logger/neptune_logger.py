import json

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure

from ...dataset_processor.colorizer import colorize_html_table
from .experiment_logger import ExperimentLogger
from ....models.ml.experiment_run_result import ExperimentRunResult
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum


class NeptuneLogger(ExperimentLogger):          
    def __init__(self, project: str, api_token: str) -> None:
        super().__init__(project)
        self.api_token = api_token
        self.run = None
    
    def __enter__(self):
        self.run = neptune.init_run(
            project=self.project,
            api_token=self.api_token,
            capture_stderr=True,
            capture_stdout=True,
            capture_traceback=True,
            capture_hardware_metrics=True,
            dependencies='infer'
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tbf):
        self.run.stop()

    def log_json(self, name: str, data: dict):
        self.run[name].upload(File.from_content(json.dumps(data), extension='json'))
    
    def log_txt(self, name: str, data: str):
        self.run[name].upload(File.from_content(data))
    
    def log_dataset(self, name: str, dataset_path: str):
        self.run[name].upload(dataset_path)
    
    def log_table(self, name: str, df: pd.DataFrame):
        self.run[name].upload(File.as_html(df))

    def get_run_result(self) -> ExperimentRunResult:
        return ExperimentRunResult(
            experiment_tracker=ExperimentTrackerEnum.Neptune,
            url=self.run.get_url(),
            run_id=self.run['sys/id'].fetch(),
        )
    
    def log_colorized_table(
        self,
        name: str,
        df: pd.DataFrame,
        matched_indices: list[tuple[int, int]],
        false_positive_indices: list[tuple[int, int]],
        false_negative_indices: list[tuple[int, int]]
    ):
        colorized_table = colorize_html_table(
            matched_indices,
            false_positive_indices,
            false_negative_indices,
            File.as_html(df).content
        )
        self.run[name].upload(File.from_content(colorized_table, extension='html'))
    
    def log_figure(self, name: str, figure: Figure):
        self.run[name].upload(figure)
    
    def log_params(self, name: str, param: dict[str]):
        self.run[name] = param

    def log_metrics(self, metrics: dict[str, int | float]) -> None:
        self.run['metrics'] = metrics
    
    def log_model_pth(self, name: str, data: bytes):
        self.run[name].upload(File.from_content(data, extension='pth'))
    
    def log_metric(self, name: str, param):
        self.run[name].append(param)
    
    def add_tags(self, tags: dict[str]) -> None:
        self.run['sys/tags'].add(list(tags.values()))
