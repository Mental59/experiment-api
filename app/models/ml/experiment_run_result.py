from typing import Optional

from pydantic import BaseModel

from .experiment_tracker_enum import ExperimentTrackerEnum


class ExperimentRunResult(BaseModel):
    experiment_tracker: ExperimentTrackerEnum
    run_id: str
    run_name: str
    project_id: str = None
    url: Optional[str] = None

    class Config:
        use_enum_values = True
