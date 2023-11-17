from uuid import uuid4

import torch


def generate_experiment_id() -> str:
    return str(uuid4())


def get_torch_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
