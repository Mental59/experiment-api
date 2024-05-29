from datetime import datetime
from app.db import models
from app.models.ml.eval_res import MetricsEvaluateRes
from app.models.ml.experiment_run_result import ExperimentRunResult
from app.services.onto.onto_parser import OntoParser


def add_experiment_from_results(
    onto_parser: OntoParser,
    run_result: ExperimentRunResult,
    params: dict,
    metrics: MetricsEvaluateRes,
    mode: str,
    user: models.UserDB,
    base_experiment_id: str | None
):
    now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    tracker_info = run_result.model_dump()
    metrics_dict = metrics.model_dump()
    params["experiment_mode"] = mode

    onto_parser.add_experiment(
        f'{run_result.run_name} ({run_result.run_id}) {now}',
        attributes=dict(
            time=now,
            tracker_info=tracker_info,
            metrics=metrics_dict,
            author=dict(id=str(user.id), login=str(user.login)),
            parameters=params
        ),
        base_experiment_id=base_experiment_id
    )
