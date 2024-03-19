from datetime import datetime

from fastapi import APIRouter, Depends

from app.db import models
from app.services.auth.auth import get_current_active_user

from ...models.ml.eval_res import MetricsEvaluateRes
from ...models.ml.ml_train_input import MLTrainExperimentInput
from ...models.ml.ml_test_input import MLTestExperimentInput
from ...services.experiment.bilstm_crf_experiments import train as train_runner, test as test_runner
from ...models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from ...models.ml.experiment_run_result import ExperimentRunResult
from ...services.onto.onto import ONTO_PARSER, ONTO_PATH

router = APIRouter(dependencies=[Depends(get_current_active_user)])


async def add_experiment(run_result: ExperimentRunResult, params: dict, metrics: MetricsEvaluateRes, mode: str):
    user: models.UserDB = await get_current_active_user()
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    tracker_info = run_result.model_dump()
    metrics_dict = metrics.model_dump()
    params["experiment_mode"] = mode

    ONTO_PARSER.add_experiment(
        f'{run_result.run_name} ({run_result.run_id}) {now}',
        attributes=dict(
            time=now,
            tracker_info=tracker_info,
            metrics=metrics_dict,
            author=dict(name=user.id),
            parameters=params
        )
    )
    ONTO_PARSER.save(ONTO_PATH)


@router.post('/train-neptune')
async def train(body: MLTrainExperimentInput, project: str, api_token: str) -> ExperimentRunResult:
    run_result, metrics, params = train_runner.run(
        project=project,
        run_name=body.run_name,
        api_token=api_token,
        dataset=body.dataset,
        embedding_dim=body.model_params.embedding_dim,
        hidden_dim=body.model_params.hidden_dim,
        batch_size=body.train_params.batch_size,
        num_epochs=body.train_params.num_epochs,
        learning_rate=body.train_params.learning_rate,
        scheduler_factor=body.train_params.scheduler_factor,
        scheduler_patience=body.train_params.scheduler_patience,
        weight_decay=body.train_params.weight_decay,
        case_sensitive=body.train_params.case_sensitive,
        test_size=body.train_params.test_size,
        num2words=body.train_params.num2words,
        experiment_tracker_type=ExperimentTrackerEnum.Neptune
    )
    await add_experiment(run_result, params, metrics, mode='train')
    return run_result


@router.post('/test-neptune')
async def test(body: MLTestExperimentInput, project: str, api_token: str) -> ExperimentRunResult:
    run_result, metrics, params = test_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        train_run_id=body.train_run_id,
        api_token=api_token,
        experiment_tracker_type=ExperimentTrackerEnum.Neptune
    )
    await add_experiment(run_result, params, metrics, mode='test')
    return run_result


@router.post('/train-mlflow')
async def train(body: MLTrainExperimentInput, project: str) -> ExperimentRunResult:
    run_result, metrics, params = train_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        embedding_dim=body.model_params.embedding_dim,
        hidden_dim=body.model_params.hidden_dim,
        batch_size=body.train_params.batch_size,
        num_epochs=body.train_params.num_epochs,
        learning_rate=body.train_params.learning_rate,
        scheduler_factor=body.train_params.scheduler_factor,
        scheduler_patience=body.train_params.scheduler_patience,
        weight_decay=body.train_params.weight_decay,
        case_sensitive=body.train_params.case_sensitive,
        test_size=body.train_params.test_size,
        num2words=body.train_params.num2words,
        experiment_tracker_type=ExperimentTrackerEnum.MLflow
    )
    await add_experiment(run_result, params, metrics, mode='train')
    return run_result


@router.post('/test-mlflow')
async def test(body: MLTestExperimentInput, project: str) -> ExperimentRunResult:
    run_result, metrics, params = test_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        train_run_id=body.train_run_id,
        experiment_tracker_type=ExperimentTrackerEnum.MLflow
    )
    await add_experiment(run_result, params, metrics, mode='test')
    return run_result
