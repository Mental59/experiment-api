from fastapi import Request
from mlflow.exceptions import MlflowException
from neptune.exceptions import NeptuneException
from fastapi.responses import JSONResponse


async def handle_exception(request: Request, call_next):
    try:
        return await call_next(request)
    except MlflowException as mlflow_exception:
        return JSONResponse(status_code=400, content=dict(message=f"MLflow Exception: {mlflow_exception}"))
    except NeptuneException as neptune_exception:
        return JSONResponse(status_code=400, content=dict(message=f"Neptune Exception: {neptune_exception}"))
    except Exception as exc:
        return JSONResponse(status_code=500, content=dict(message=f"Unhandled Exception: {exc}"))
