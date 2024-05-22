import transformers

from app.core.decorators import single_experiment_run
from app.models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from app.services.experiment.tracker.utils import get_experiment_tracker
from app.services.dataset_processor import generator as dataset_generator
from app.constants.save_keys import *
from app.services.experiment.transformers.utils import evaluate_transformer_pipeline


@single_experiment_run
def run_by_model_name_or_path(
    project: str,
    run_name: str,
    model_name_or_path: str,
    dataset: str,
    experiment_tracker_type: ExperimentTrackerEnum,
    task: str = 'ner',
    batch_size: int | None = None,
    num_workers: int | None = None,
    **kwargs
):
    with get_experiment_tracker(experiment_tracker_type, project, run_name, **kwargs) as experiment_tracker:
        sents = dataset_generator.get_sents_from_dataset(dataset, case_sensitive=True)
        experiment_tracker.log_dataset(DATASET_SAVE_KEY, dataset_generator.get_dataset_path(dataset))

        params = {
            'model_name': model_name_or_path,
            'dataset': dataset,
            'transformer_experiment': True
        }
        experiment_tracker.log_params(PARAMETERS_SAVE_KEY, params)
        experiment_tracker.add_tags(
            dict(
                model_name=model_name_or_path,
                mode='test',
                run_name=run_name,
                transformer_experiment='True'
            )
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        model = transformers.AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        pipeline = transformers.pipeline(task=task, model=model, tokenizer=tokenizer)

        eval_res, outputs = evaluate_transformer_pipeline(pipeline, sents, batch_size=batch_size, num_workers=num_workers)

        experiment_tracker.log_metrics(eval_res.metrics.model_dump())
        experiment_tracker.log_json(UNK_FOREACH_TAG_SAVE_KEY, eval_res.unk_foreach_tag)
        experiment_tracker.log_json(TRANSFORMER_RAW_OUTPUT_KEY, outputs)
        experiment_tracker.log_txt(FLAT_CLASSIFICATION_REPORT_SAVE_KEY, eval_res.flat_classification_report)
        experiment_tracker.log_figure(DIAGRAM_SAVE_KEY, eval_res.fig)
        experiment_tracker.log_colorized_table(
            PREDICTED_DF_SAVE_KEY,
            eval_res.df_predicted,
            eval_res.matched_indices,
            eval_res.false_positive_indices,
            eval_res.false_negative_indices
        )
        experiment_tracker.log_table(ACTUAL_DF_SAVE_KEY, eval_res.df_actual)

        return experiment_tracker.get_run_result(), eval_res.metrics, {}


def run_by_train_experiment():
    pass
