from abc import ABCMeta, abstractmethod


class RunLoader(metaclass=ABCMeta):
    def __init__(self, project: str, run_id: str):
        self.project = project
        self.run_id = run_id
    
    @abstractmethod
    def get_params(self, save_key: str) -> dict[str, str | int | float | bool]:
        pass
    
    @abstractmethod
    def get_model_state_dict(self, save_key: str):
        pass

    @abstractmethod
    def get_word_to_ix(self, save_key: str) -> dict[str, int]:
        pass

    @abstractmethod
    def get_tag_to_ix(self, save_key: str) -> dict[str, int]:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass
