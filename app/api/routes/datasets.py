from typing import Annotated

from fastapi import APIRouter, UploadFile, File

from ...services.dataset_processor import loader

router = APIRouter()


@router.get("/")
async def get_datasets() -> list[str]:
    return loader.get_datasets()


@router.post("/upload")
async def upload_datasets(datasets: Annotated[list[UploadFile], File(description="Multiple datasets")]) -> list[str]:
    await loader.upload_datasets(datasets)
    return loader.get_datasets()
