from fastapi import APIRouter

from ...models.ml.ml_train_input import MLTrainExperimentInput
from ...models.ml.ml_test_input import MLTestExperimentInput
from ...services.experiment.bilstm_crf_experiments import train as train_runner, test as test_runner
from ...models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from ...models.ml.experiment_run_result import ExperimentRunResult

router = APIRouter()


@router.post('/train-neptune')
def train(body: MLTrainExperimentInput, project: str, api_token: str) -> ExperimentRunResult:
    return train_runner.run(
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


@router.post('/test-neptune')
def test(body: MLTestExperimentInput, project: str, api_token: str) -> ExperimentRunResult:
    return test_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        train_run_id=body.train_run_id,
        api_token=api_token,
        experiment_tracker_type=ExperimentTrackerEnum.Neptune
    )


@router.post('/train-mlflow')
def train(body: MLTrainExperimentInput, project: str) -> ExperimentRunResult:
    return train_runner.run(
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


@router.post('/test-mlflow')
def test(body: MLTestExperimentInput, project: str) -> ExperimentRunResult:
    return test_runner.run(
        project=project,
        run_name=body.run_name,
        dataset=body.dataset,
        train_run_id=body.train_run_id,
        experiment_tracker_type=ExperimentTrackerEnum.MLflow
    )
