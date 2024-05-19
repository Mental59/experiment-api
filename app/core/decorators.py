from functools import wraps

from fastapi import HTTPException

from app.core.exceptions import create_exception_details


IS_EXPERIMENT_RUNNING = False

def single_experiment_run(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global IS_EXPERIMENT_RUNNING
        if IS_EXPERIMENT_RUNNING:
            raise HTTPException(status_code=400, detail=create_exception_details(message="Эксперимент уже запущен"))
        IS_EXPERIMENT_RUNNING = True
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            IS_EXPERIMENT_RUNNING = False
    return wrapper
