from fastapi import FastAPI, UploadFile

from .services import startup as startup_service
from .services.dataset_processor import uploader

app = FastAPI()
startup_service.on_startup()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/dataset/upload")
async def upload_dataset(dataset: UploadFile):
    await uploader.upload_dataset(dataset)
