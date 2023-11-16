from fastapi import APIRouter

from ...models.ml.ml_train import MLTrainExperimentInput

router = APIRouter()


@router.post("/train")
def train(params: MLTrainExperimentInput):
    return params
