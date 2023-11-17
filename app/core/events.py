from typing import Callable

from ..services.file_manager import file_manager


def create_start_app_handler() -> Callable:
    def start_app():
        file_manager.create_artifact_folders()
       
    return start_app


def create_stop_app_handler() -> Callable:
    def stop_app():
        pass

    return stop_app
