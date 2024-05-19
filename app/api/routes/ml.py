from typing import Annotated

from fastapi import APIRouter, Depends

from app.db import models
from app.services.auth.auth import get_current_active_user
from app.services.onto.utils import add_experiment_from_results

from ...models.ml.ml_train_input import MLTrainExperimentInput
from ...models.ml.ml_test_input import MLTestExperimentInput
from ...services.experiment.bilstm_crf_experiments import train as train_bilstm_crf_runner, test as test_bilstm_crf_runner
from ...models.ml.experiment_run_result import ExperimentRunResult

router = APIRouter(dependencies=[Depends(get_current_active_user)])


@router.post('/train-bilstm-crf')
def train(
    body: MLTrainExperimentInput,
    project: str,
    api_token: str | None,
    current_user: Annotated[models.UserDB, Depends(get_current_active_user)],
) -> ExperimentRunResult:
    run_result, metrics, params = train_bilstm_crf_runner.run(
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
        experiment_tracker_type=body.experiment_tracker,
    )

    add_experiment_from_results(
        run_result,
        params,
        metrics,
        mode='train',
        user=current_user,
        base_experiment_id=body.base_experiment_id
    )

    return run_result


@router.post('/test-bilstm-crf')
def test(
    body: MLTestExperimentInput,
    project: str,
    api_token: str | None,
    current_user: Annotated[models.UserDB, Depends(get_current_active_user)],
) -> ExperimentRunResult:
    run_result, metrics, params = test_bilstm_crf_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        train_run_id=body.train_run_id,
        api_token=api_token,
        experiment_tracker_type=body.experiment_tracker
    )

    add_experiment_from_results(
        run_result,
        params,
        metrics,
        mode='test',
        user=current_user,
        base_experiment_id=body.base_experiment_id
    )

    return run_result
