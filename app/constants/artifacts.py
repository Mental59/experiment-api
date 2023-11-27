import os

ARTIFACTS_PATH = 'artifacts'


class PATHS:
    FEATURES_PATH = os.path.join(ARTIFACTS_PATH, 'features')
    EXPERIMENTS_PATH = os.path.join(ARTIFACTS_PATH, 'experiments')
    DATASETS_PATH = os.path.join(ARTIFACTS_PATH, 'datasets')
    TEMP_PATH = os.path.join(ARTIFACTS_PATH, 'temp')

    @classmethod
    def get_paths(cls) -> list[str]:
        keys = [key for key in cls.__dict__ if not key.startswith('__')]
        paths = [cls.__dict__[key] for key in keys]
        paths = [path for path in paths if isinstance(path, str)]
        return paths
