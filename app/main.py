from fastapi import FastAPI

from .core.events import create_start_app_handler, create_stop_app_handler
from .api.routes import api


def get_application() -> FastAPI:
    app = FastAPI()

    app.add_event_handler('startup', create_start_app_handler())
    app.add_event_handler('shutdown', create_stop_app_handler())

    app.include_router(api.router)

    return app


app = get_application()
