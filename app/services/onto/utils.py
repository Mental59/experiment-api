from datetime import datetime

from fastapi import HTTPException, UploadFile
from app.core.exceptions import create_exception_details
from app.db import models
from app.models.ml.eval_res import MetricsEvaluateRes
from app.models.ml.experiment_run_result import ExperimentRunResult


def add_experiment_from_results(
    onto_parser,
    run_result: ExperimentRunResult,
    params: dict,
    metrics: MetricsEvaluateRes,
    mode: str,
    user: models.UserDB,
    base_experiment_id: str | None
):
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    tracker_info = run_result.model_dump()
    metrics_dict = metrics.model_dump()
    params["experiment_mode"] = mode

    onto_parser.add_experiment(
        f'{run_result.run_name} ({run_result.run_id}) {now}',
        attributes=dict(
            time=now,
            tracker_info=tracker_info,
            metrics=metrics_dict,
            author=dict(id=str(user.id), login=str(user.login)),
            parameters=params
        ),
        base_experiment_id=base_experiment_id
    )


def decode_bytes(content_bytes: bytes):
    is_failed = False
    text = None
    errors = []

    try:
        text = content_bytes.decode("utf-8")
    except ValueError as error:
        errors.append(error)
        is_failed = True
    
    if is_failed:
        try:
            text = content_bytes.decode("utf-16")
        except ValueError as error:
            errors.append(error)
            is_failed = True

    return text, errors


async def read_uploaded_file(file: UploadFile):
    content_bytes = await file.read()
    text, errors = decode_bytes(content_bytes)
    if text is None:
        raise HTTPException(400, detail=create_exception_details(f"Ошибка при чтении файла {file.filename}; Причина: {errors}"))
    return text
