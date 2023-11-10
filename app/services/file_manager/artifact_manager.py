import os
from ...constants.artifacts import FEATURES_PATH


def create_artifact_folders():
  os.makedirs(FEATURES_PATH, exist_ok=True)
