from typing import Literal
import os

ARTIFACTS_PATH = 'artifacts'

PATHS: dict[Literal['FEATURES_PATH', 'EXPERIMENTS_PATH', 'DATASETS_PATH', 'TEMP_PATH'], str] = dict(
    FEATURES_PATH=os.path.join(ARTIFACTS_PATH, 'features'),
    EXPERIMENTS_PATH=os.path.join(ARTIFACTS_PATH, 'experiments'),
    DATASETS_PATH=os.path.join(ARTIFACTS_PATH, 'datasets'),
    TEMP_PATH=os.path.join(ARTIFACTS_PATH, 'temp')
)
