from abc import ABCMeta, abstractmethod

import pandas as pd
from matplotlib.figure import Figure


class ExperimentLogger(metaclass=ABCMeta):
    def __init__(self, project: str):
        self.project = project
    
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
    def log_param(self, name: str, param) -> None:
        pass
    
    @abstractmethod
    def log_binary(self, name: str, data: bytes, extension: str) -> None:
        pass
    
    @abstractmethod
    def append_param(self, name: str, param) -> None:
        pass
    
    @abstractmethod
    def add_tags(self, tags: dict[str]) -> None:
        pass