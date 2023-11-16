import os

from ...constants.artifacts import PATHS
from ...constants.core import NOTEBOOKS_PATH


def create_artifact_folders():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)


def create_core_folders():
    os.makedirs(NOTEBOOKS_PATH, exist_ok=True)
