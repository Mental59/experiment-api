import os
import shutil

from ...constants.artifacts import PATHS


def create_artifact_folders():
    for path in PATHS.get_paths():
        os.makedirs(path, exist_ok=True)


def clear_temp_dir():
    if os.path.exists(PATHS.TEMP_PATH):
        shutil.rmtree(PATHS.TEMP_PATH)
    os.makedirs(PATHS.TEMP_PATH, exist_ok=True)
