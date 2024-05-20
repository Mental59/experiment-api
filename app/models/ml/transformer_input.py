from app.models.ml.ml_base_input import MLBaseInput


class TestTransformerByModelName(MLBaseInput):
    model_name_or_path: str
    task: str = 'ner'
    batch_size: int | None = None
    num_workers: int | None = None
