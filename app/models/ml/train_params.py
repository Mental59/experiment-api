from pydantic import BaseModel


class TrainParams(BaseModel):
    batch_size: int = 2048
    num_epochs: int = 40
    learning_rate: float = 1e-2
    scheduler_factor: float = 1e-1
    scheduler_patience: int = 10
    weight_decay: float = 1e-4
    case_sensitive_vocab: bool = False
    test_size: float = 0.2
    num2words: bool = True
