from fastapi import APIRouter

from ...models.ml.ml_train import MLTrainExperimentInput
from ...services.experiment.bilstm_crf_experiments import train_neptune

router = APIRouter()


@router.post("/train-neptune")
def train(params: MLTrainExperimentInput, project: str, api_token: str):
    train_neptune.run(
        project=project,
        run_name=params.run_name,
        api_token=api_token,
        dataset=params.dataset,
        embedding_dim=params.model_params.embedding_dim,
        hidden_dim=params.model_params.hidden_dim,
        batch_size=params.train_params.batch_size,
        num_epochs=params.train_params.num_epochs,
        learning_rate=params.train_params.learning_rate,
        scheduler_factor=params.train_params.scheduler_factor,
        scheduler_patience=params.train_params.scheduler_patience,
        weight_decay=params.train_params.weight_decay,
        case_sensitive=params.train_params.case_sensitive,
        test_size=params.train_params.test_size,
        num2words=params.train_params.num2words,
    )
