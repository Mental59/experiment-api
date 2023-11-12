import os

from fastapi import UploadFile

from ...constants.artifacts import PATHS
from ..file_manager.file import write_file


def upload_dataset(file: UploadFile):
  return write_file(os.path.join(PATHS['DATASETS_PATH'], file.filename), file)