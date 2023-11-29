import os
import shutil

from ...constants.artifacts import PATHS


class TempFolder:
    def __init__(self, folder_name: str):
        self.path = os.path.join(PATHS.TEMP_PATH, folder_name)

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)

