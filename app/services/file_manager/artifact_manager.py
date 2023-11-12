import os
from ...constants.artifacts import PATHS


def create_artifact_folders():
  for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
