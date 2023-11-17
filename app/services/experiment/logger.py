import os
import json

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure

from ...constants.artifacts import PATHS


def log_json_neptune(run: neptune.Run, data: dict, neptune_path: str) -> None:
    run[neptune_path].upload(File.from_content(json.dumps(data), extension='json'))


def log_txt_neptune(run: neptune.Run, data: str, neptune_path: str) -> None:
    run[neptune_path].upload(File.from_content(data))


def log_by_path_neptune(run: neptune.Run, neptune_path: str, data_path: str):
    run[neptune_path].upload(data_path)


def log_table_neptune(run: neptune.Run, neptune_path: str, df: pd.DataFrame):
    run[neptune_path].upload(File.as_html(df))


def log_figure_neptune(run: neptune.Run, neptune_path: str, figure: Figure):
    run[neptune_path].upload(figure)
