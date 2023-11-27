import os
import json

import neptune
import torch

from ....constants.artifacts import PATHS
from .run_loader import RunLoader


class NeptuneRunLoader(RunLoader):
    def __init__(self, project: str, run_id: str, api_token: str) -> None:
        super().__init__(project=project, run_id=run_id)
        self.run = neptune.init_run(with_id=run_id, mode='read-only', api_token=api_token, project=project)

    def get_params(self) -> dict[str, str | int | float | bool]:
        return self.run['parameters'].fetch()
    
    def get_model_state_dict(self):
        self.run['model_checkpoints/best_model'].download(destination=PATHS.TEMP_PATH)
        return torch.load(os.path.join(PATHS.TEMP_PATH, 'best_model.pth'))

    def get_word_to_ix(self) -> dict[str, int]:
        self.run['data/word_to_ix'].download(destination=PATHS.TEMP_PATH)
        with open(os.path.join(PATHS.TEMP_PATH, 'word_to_ix.json'), encoding='utf-8') as file:
            return json.load(file)

    def get_tag_to_ix(self) -> dict[str, int]:
        self.run['data/tag_to_ix'].download(destination=PATHS.TEMP_PATH)
        with open(os.path.join(PATHS.TEMP_PATH, 'tag_to_ix.json'), encoding='utf-8') as file:
            return json.load(file)
