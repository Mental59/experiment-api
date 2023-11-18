import json
from typing import Iterable

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure

from ..dataset_processor.colorizer import colorize_html_table


class NeptuneLogger:          
    def __init__(self, project: str, api_token: str) -> None:
        self.project = project
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
    
    def log_by_path(self, name: str, path: str):
        self.run[name].upload(path)
    
    def log_table(self, name: str, df: pd.DataFrame):
        self.run[name].upload(File.as_html(df))
    
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
    
    def log_param(self, name: str, param):
        self.run[name] = param
    
    def log_binary(self, name: str, data: bytes, extension: str):
        self.run[name].upload(File.from_content(data, extension=extension))
    
    def append_param(self, name: str, param):
        self.run[name].append(param)
    
    def add_tags(self, tags: str | Iterable[str]):
        self.run['sys/tags'].add(tags)
