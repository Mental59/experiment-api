import os
import json

import mlflow
import torch

from .run_loader import RunLoader
from ....services.file_manager.temp_folder import TempFolder


class MLFlowRunLoader(RunLoader):
    def __init__(self, project: str, run_id: str):
        super().__init__(project=project, run_id=run_id)
        self.experiment = mlflow.get_experiment_by_name(project)
        self.run = mlflow.get_run(run_id)

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
        with TempFolder(self.run.info.run_id) as temp_folder:
            mlflow.artifacts.download_artifacts(
                run_id=self.run.info.run_id,
                artifact_path=f'{save_key}.pth',
                dst_path=temp_folder.path
            )
            return torch.load(os.path.join(temp_folder.path, *save_key.split('/')) + '.pth')

    def get_word_to_ix(self, save_key: str) -> dict[str, int]:
        return self.__load_json(save_key)

    def get_tag_to_ix(self, save_key: str) -> dict[str, int]:
        return self.__load_json(save_key)

    def __load_json(self, save_key: str):
        content = mlflow.artifacts.load_text(self.run.info.artifact_uri + f'/{save_key}.json')
        return json.loads(content)

    def stop(self) -> None:
        pass

    @staticmethod
    def __is_float(s: str):
        try:
            float(s)
            return True
        except ValueError:
            return False
