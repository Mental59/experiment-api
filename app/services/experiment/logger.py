import json
from typing import Iterable

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure


class NeptuneLogger:          
    def __init__(self, project: str, api_token: str) -> None:
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
            capture_stderr=True,
            capture_stdout=True,
            capture_traceback=True,
            capture_hardware_metrics=True,
            dependencies='infer'
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tbf):
        self.run.stop()

    def json(self, name: str, data: dict):
        self.run[name].upload(File.from_content(json.dumps(data), extension='json'))
    
    def txt(self, name: str, data: str):
        self.run[name].upload(File.from_content(data))
    
    def by_path(self, name: str, path: str):
        self.run[name].upload(path)
    
    def table(self, name: str, df: pd.DataFrame):
        self.run[name].upload(File.as_html(df))
    
    def figure(self, name: str, figure: Figure):
        self.run[name].upload(figure)
    
    def param(self, name: str, param):
        self.run[name] = param
    
    def binary(self, name: str, data: bytes, extension: str):
        self.run[name].upload(File.from_content(data, extension=extension))
    
    def append_param(self, name: str, param):
        self.run[name].append(param)
    
    def add_tags(self, tags: str | Iterable[str]):
        self.run['sys/tags'].add(tags)
