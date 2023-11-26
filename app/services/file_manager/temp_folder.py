import os
import shutil


class TempFolder:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)

