from pydantic import BaseModel
import pandas as pd
from matplotlib.figure import Figure


class MetricsEvaluateRes(BaseModel):
    f1_weighted: float
    precision_weighted: float
    recall_weighted: float
    accuracy: float
    confidence: float


class EvaluateRes(BaseModel):
    unk_foreach_tag: dict[str, float]
    metrics: MetricsEvaluateRes
    flat_classification_report: str
    df_predicted: pd.DataFrame
    df_actual: pd.DataFrame
    fig: Figure
    matched_indices: list[tuple[int, int]]
    false_positive_indices: list[tuple[int, int]]
    false_negative_indices: list[tuple[int, int]]

    class Config:
        arbitrary_types_allowed=True
