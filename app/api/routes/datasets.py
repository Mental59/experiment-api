from fastapi import APIRouter, UploadFile

from ...services.dataset_processor import uploader

router = APIRouter()


@router.post("/upload")
async def upload_dataset(dataset: UploadFile):
    await uploader.upload_dataset(dataset)
