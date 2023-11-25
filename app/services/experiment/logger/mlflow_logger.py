from typing import Iterable

import pandas as pd
from matplotlib.figure import Figure
import mlflow

from .experiment_logger import ExperimentLogger


class MLFlowLogger(ExperimentLogger):
    def __init__(self, project: str):
        super().__init__(project)
    
    def __enter__(self):
        mlflow.set_experiment(self.project)
        mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tbf):
        mlflow.end_run()
    
    def log_json(self, name: str, data: dict) -> None:
        pass
    
    def log_txt(self, name: str, data: str) -> None:
        pass
    
    def log_by_path(self, name: str, path: str) -> None:
        pass
    
    def log_table(self, name: str, df: pd.DataFrame) -> None:
        pass
    
    def log_colorized_table(
        self,
        name: str,
        df: pd.DataFrame,
        matched_indices: list[tuple[int, int]],
        false_positive_indices: list[tuple[int, int]],
        false_negative_indices: list[tuple[int, int]]
    ) -> None:
        pass
    
    def log_figure(self, name: str, figure: Figure) -> None:
        pass
    
    def log_param(self, name: str, param) -> None:
        pass
    
    def log_binary(self, name: str, data: bytes, extension: str) -> None:
        pass
    
    def append_param(self, name: str, param) -> None:
        pass
    
    def add_tags(self, tags: str | Iterable[str]) -> None:
        pass
