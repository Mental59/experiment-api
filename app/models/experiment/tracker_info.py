from enum import Enum

from pydantic import BaseModel

from ...models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from ...constants.nn import TRAIN_MODE_NAME, TEST_MODE_NAME

class RunType(Enum):
    Train=TRAIN_MODE_NAME
    Test=TEST_MODE_NAME
    Unknown='unknown'


class RunInfo(BaseModel):
    run_id: str
    run_name: str
    run_type: RunType

    class Config:
        use_enum_values = True


class ExperimentInfo(BaseModel):
    project_id: str
    project_name: str
    runs: list[RunInfo]


class ExperimentTrackerInfo(BaseModel):
    tracker: ExperimentTrackerEnum
    projects: list[ExperimentInfo]

    class Config:
        use_enum_values = True
