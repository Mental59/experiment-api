from typing import Annotated

from fastapi import APIRouter, UploadFile, File

from ...services.dataset_processor import uploader
from ...services.file_manager import file_manager
from ...constants.artifacts import PATHS

router = APIRouter()


@router.get("/")
async def get_datasets() -> list[str]:
    return file_manager.get_datasets()


@router.post("/upload")
async def upload_datasets(datasets: Annotated[list[UploadFile], File(description="Multiple datasets")]) -> None:
    await uploader.upload_datasets(datasets)
