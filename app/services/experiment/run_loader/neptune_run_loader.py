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

    def get_params(self, save_key: str) -> dict[str, str | int | float | bool]:
        return self.run[save_key].fetch()
    
    def get_model_state_dict(self, save_key: str):
        self.run[save_key].download(destination=PATHS.TEMP_PATH)
        return torch.load(os.path.join(PATHS.TEMP_PATH, save_key.split('/')[-1] + '.pth'))

    def get_word_to_ix(self, save_key: str) -> dict[str, int]:
        self.run[save_key].download(destination=PATHS.TEMP_PATH)
        with open(os.path.join(PATHS.TEMP_PATH, save_key.split('/')[-1] + '.json'), encoding='utf-8') as file:
            return json.load(file)

    def get_tag_to_ix(self, save_key: str) -> dict[str, int]:
        self.run[save_key].download(destination=PATHS.TEMP_PATH)
        with open(os.path.join(PATHS.TEMP_PATH, save_key.split('/')[-1] + '.json'), encoding='utf-8') as file:
            return json.load(file)
