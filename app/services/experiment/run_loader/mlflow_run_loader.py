from .run_loader import RunLoader


class MLFlowRunLoader(RunLoader):
    def __init__(self, project: str, run_id: str):
        super().__init__(project=project, run_id=run_id)

    def get_params(self) -> dict[str, str | int | float | bool]:
        pass

    def get_model_state_dict(self):
        pass

    def get_word_to_ix(self) -> dict[str, int]:
        pass

    def get_tag_to_ix(self) -> dict[str, int]:
        pass
