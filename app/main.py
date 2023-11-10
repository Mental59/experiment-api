from fastapi import FastAPI

from .services import startup as startup_service

app = FastAPI()
startup_service.on_startup()

@app.get("/")
def read_root():
    return {"Hello": "World"}
