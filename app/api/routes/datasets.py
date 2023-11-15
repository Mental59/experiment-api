import os
from typing import Annotated

from fastapi import APIRouter, UploadFile, File

from ...services.dataset_processor import uploader
from ...constants.artifacts import PATHS

router = APIRouter()


@router.post("/upload")
async def upload_datasets(datasets: Annotated[list[UploadFile], File(description="Multiple datasets")]) -> None:
    await uploader.upload_datasets(datasets)


@router.get("/")
async def get_datasets() -> list[str]:
    return os.listdir(PATHS["DATASETS_PATH"])
