import os
import json

import neptune
from neptune.types import File
import pandas as pd
from matplotlib.figure import Figure

from ...constants.artifacts import PATHS


def log_json_neptune(run: neptune.Run, data: dict, neptune_path: str, filename: str) -> None:
    data_path = get_path(filename)
    with open(data_path, 'w', encoding='utf-8') as file:
        json.dump(data, file)
    run[neptune_path].upload(data_path)


def log_txt_neptune(run: neptune.Run, data: str, neptune_path: str, filename: str) -> None:
    data_path = get_path(filename)
    with open(data_path, 'w', encoding='utf-8') as file:
        file.write(data)
    run[neptune_path].upload(data_path)


def log_by_path_neptune(run: neptune.Run, neptune_path: str, data_path: str):
    run[neptune_path].upload(data_path)


def log_table_neptune(run: neptune.Run, neptune_path: str, df: pd.DataFrame):
    run[neptune_path].upload(File.as_html(df))


def log_figure_neptune(run: neptune.Run, neptune_path: str, figure: Figure, filename: str):
    figure_path = get_path(filename)

    figure.savefig(figure_path)
    run[neptune_path].upload(figure_path)


def get_path(filename: str) -> str:
    return os.path.join(PATHS['TEMP_PATH'], filename)
