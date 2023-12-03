from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.events import create_start_app_handler, create_stop_app_handler
from .core.handlers import handle_exception
from .api.routes import api


def get_application() -> FastAPI:
    app = FastAPI()

    app.add_event_handler('startup', create_start_app_handler())
    app.add_event_handler('shutdown', create_stop_app_handler())

    app.middleware('http')(handle_exception)
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

    app.include_router(api.router)

    return app


app = get_application()
