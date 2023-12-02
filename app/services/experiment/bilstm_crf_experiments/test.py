from torch.utils.data import DataLoader

from ....services.nn.utils import CustomDataset, evaluate_model
from ....services.nn.bilstm_crf import BiLSTM_CRF
from ....services.dataset_processor import generator as dataset_generator
from ....services.dataset_processor import analyzer as dataset_analyzer
from ....services.experiment import setupper as experiment_setupper
from ....services.experiment.logger.utils import get_experiment_tracker
from ....services.experiment.run_loader.utils import get_run_loader
from ....constants.nn import PAD, UNK
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from ....constants.save_keys import *


def run(
    project: str,
    run_name: str,
    dataset: str,
    train_run_id: str,
    experiment_tracker: ExperimentTrackerEnum,
    **kwargs
):
    with get_experiment_tracker(experiment_tracker, project, run_name, **kwargs) as experiment_logger:
        device = experiment_setupper.get_torch_device()

        train_run = get_run_loader(experiment_tracker=experiment_tracker, project=project, run_id=train_run_id, **kwargs)
        train_run_params = train_run.get_params(PARAMETERS_SAVE_KEY)
        word_to_ix = train_run.get_word_to_ix(WORD_TO_IX_SAVE_KEY)
        tag_to_ix = train_run.get_tag_to_ix(TAG_TO_IX_SAVE_KEY)
        model_state_dict = train_run.get_model_state_dict(BEST_MODEL_SAVE_KEY)

        sents = dataset_generator.get_sents_from_dataset(dataset, case_sensitive=train_run_params['case_sensitive'])
        experiment_logger.log_dataset(DATASET_SAVE_KEY, dataset_generator.get_dataset_path(dataset))
        
        unknown_labels = dataset_analyzer.sents_has_unknown_labels(sents, tag_to_ix)
        ix_to_tag = dataset_generator.generate_ix_to_key(tag_to_ix)
        labels = dataset_generator.generate_labels(tag_to_ix)

        params = {
            'train_run_id': train_run_id,
            'dataset': dataset,
            'device': device,
            'unknown_labels': unknown_labels
        }
        experiment_logger.log_params(PARAMETERS_SAVE_KEY, params)
        experiment_logger.log_params(TRAIN_PARAMETERS_SAVE_KEY, train_run_params)
        experiment_logger.add_tags(dict(model_name=train_run_params['model_name'], mode='test', run_name=run_name))

        vocab_size = len(word_to_ix)
        num_tags = len(tag_to_ix)
        embedding_dim = train_run_params['embedding_dim']
        hidden_dim = train_run_params['hidden_dim']

        dataset = CustomDataset(sents, tag_to_ix, word_to_ix, convert_nums2words=train_run_params['num2words'])
        dataloader = DataLoader(dataset, batch_size=train_run_params['batch_size'], shuffle=False, drop_last=False)

        model = BiLSTM_CRF(vocab_size, num_tags, embedding_dim, hidden_dim, word_to_ix[PAD]).to(device).eval()
        model.load_state_dict(model_state_dict)

        eval_res = evaluate_model(
            model=model,
            dataset=dataset,
            dataloader=dataloader,
            pad_idx=word_to_ix[PAD],
            device=device,
            ix_to_tag=ix_to_tag,
            unk_idx=word_to_ix[UNK],
            labels=labels
        )
        
        experiment_logger.log_metrics(eval_res.metrics.model_dump())
        experiment_logger.log_json(UNK_FOREACH_TAG_SAVE_KEY, eval_res.unk_foreach_tag)
        experiment_logger.log_txt(FLAT_CLASSIFICATION_REPORT_SAVE_KEY, eval_res.flat_classification_report)
        experiment_logger.log_figure(DIAGRAM_SAVE_KEY, eval_res.fig)
        experiment_logger.log_colorized_table(
            PREDICTED_DF_SAVE_KEY,
            eval_res.df_predicted,
            eval_res.matched_indices,
            eval_res.false_positive_indices,
            eval_res.false_negative_indices
        )
        experiment_logger.log_table(ACTUAL_DF_SAVE_KEY, eval_res.df_actual)

        return experiment_logger.get_run_result()
