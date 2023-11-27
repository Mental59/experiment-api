import os
import json

import mlflow
import torch
from fastapi import HTTPException

from .run_loader import RunLoader


class MLFlowRunLoader(RunLoader):
    def __init__(self, project: str, run_id: str):
        super().__init__(project=project, run_id=run_id)
        self.experiment = mlflow.get_experiment_by_name(project)
        self.run = mlflow.get_run(run_id)

        self.run_location = os.path.join('mlruns', self.experiment.experiment_id, run_id)
        self.artifacts_location = os.path.join(self.run_location, 'artifacts')

        self.check_folders()

    def get_params(self, save_key: str) -> dict[str, str | int | float | bool]:
        params = {}
        run_params = self.run.data.params
        for key in filter(lambda _key: _key.startswith(f'{save_key}/'), run_params):
            param_name = key.replace(f'{save_key}/', '')
            param_value = run_params[key]

            if param_value == 'True':
                param_value = True
            elif param_value == 'False':
                param_value = False
            elif param_value.isnumeric():
                param_value = int(param_value)
            elif self.__is_float(param_value):
                param_value = float(param_value)

            params[param_name] = param_value

        return params

    def get_model_state_dict(self, save_key: str):
        path = os.path.join(self.artifacts_location, *save_key.split('/')) + '.pth'
        return torch.load(path)

    def get_word_to_ix(self, save_key: str) -> dict[str, int]:
        path = os.path.join(self.artifacts_location, *save_key.split('/')) + '.json'
        return self.__load_json(path)

    def get_tag_to_ix(self, save_key: str) -> dict[str, int]:
        path = os.path.join(self.artifacts_location, *save_key.split('/')) + '.json'
        return self.__load_json(path)

    @staticmethod
    def __load_json(path: str):
        with open(path) as file:
            return json.load(file)

    @staticmethod
    def __is_float(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def check_folders(self):
        if not os.path.exists(self.run_location):
            raise HTTPException(status_code=400, detail=dict(message=f'Run location {self.run_location} does not exist'))

        if not os.path.exists(self.artifacts_location):
            raise HTTPException(status_code=400, detail=dict(message='No artifacts in the run'))
