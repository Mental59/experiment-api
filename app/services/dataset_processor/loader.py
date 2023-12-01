import os
import asyncio

from fastapi import UploadFile

from ...constants.artifacts import PATHS
from ..file_manager.file import write_file


async def upload_dataset(file: UploadFile) -> None:
    await write_file(os.path.join(PATHS.DATASETS_PATH, file.filename), file)


async def upload_datasets(files: list[UploadFile]):
    await asyncio.gather(*[upload_dataset(file) for file in files])


def get_datasets() -> list[str]:
    return [os.path.splitext(name)[0] for name in os.listdir(PATHS.DATASETS_PATH)]
