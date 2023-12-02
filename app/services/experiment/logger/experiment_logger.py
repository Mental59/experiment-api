from abc import ABCMeta, abstractmethod

import pandas as pd
from matplotlib.figure import Figure

from ....models.ml.experiment_run_result import ExperimentRunResult


class ExperimentLogger(metaclass=ABCMeta):
    def __init__(self, project: str, run_name: str):
        self.project = project
        self.run_name = run_name
    
    @abstractmethod
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tbf):
        pass
    
    @abstractmethod
    def log_json(self, name: str, data: dict) -> None:
        pass
    
    @abstractmethod
    def log_txt(self, name: str, data: str) -> None:
        pass
    
    @abstractmethod
    def log_dataset(self, name: str, dataset_path: str) -> None:
        pass
    
    @abstractmethod
    def log_table(self, name: str, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def get_run_result(self) -> ExperimentRunResult:
        pass
    
    @abstractmethod
    def log_colorized_table(
        self,
        name: str,
        df: pd.DataFrame,
        matched_indices: list[tuple[int, int]],
        false_positive_indices: list[tuple[int, int]],
        false_negative_indices: list[tuple[int, int]]
    ) -> None:
        pass
    
    @abstractmethod
    def log_figure(self, name: str, figure: Figure) -> None:
        pass
    
    @abstractmethod
    def log_params(self, name: str, param: dict[str]) -> None:
        pass
    
    @abstractmethod
    def log_model_pth(self, name: str, data: bytes) -> None:
        pass
    
    @abstractmethod
    def log_metric(self, name: str, param) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, int | float]) -> None:
        pass
    
    @abstractmethod
    def add_tags(self, tags: dict[str]) -> None:
        pass
