import os

from ...constants.artifacts import PATHS
from ...constants.core import NOTEBOOKS_PATH


def create_artifact_folders():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)


def create_core_folders():
    os.makedirs(NOTEBOOKS_PATH, exist_ok=True)


def get_datasets():
    return [os.path.splitext(name)[0] for name in os.listdir(PATHS["DATASETS_PATH"])]
