from torch.utils.data import DataLoader

from ....services.nn.utils import CustomDataset, evaluate_model
from ....services.nn.bilstm_crf import BiLSTM_CRF
from ....services.dataset_processor import generator as dataset_generator
from ....services.dataset_processor import analyzer as dataset_analyzer
from ....services.experiment import setupper as experiment_setupper
from ....services.experiment.logger import NeptuneLogger
from ....services.experiment.run_loader import NeptuneRunLoader
from ....constants.nn import PAD, UNK


def run(
    project: str,
    run_name: str,
    api_token: str,
    dataset: str,
    train_run_id: str
):
    with NeptuneLogger(project, api_token) as neptune_logger:
        device = experiment_setupper.get_torch_device()

        train_run = NeptuneRunLoader(project, api_token, run_id=train_run_id)
        train_run_params = train_run.get_params()
        word_to_ix = train_run.get_word_to_ix()
        tag_to_ix = train_run.get_tag_to_ix()
        model_state_dict = train_run.get_model_state_dict()

        sents = dataset_generator.get_sents_from_dataset(dataset, case_sensitive=train_run_params['case_sensitive'])
        unknown_labels = dataset_analyzer.sents_has_unknown_labels(sents, tag_to_ix)
        ix_to_tag = dataset_generator.generate_ix_to_key(tag_to_ix)
        ix_to_word = dataset_generator.generate_ix_to_key(word_to_ix)
        labels = dataset_generator.generate_labels(tag_to_ix)

        params = {
            'train_run_id': train_run_id,
            'dataset': dataset,
            'device': device,
            'unknown_labels': unknown_labels
        }
        neptune_logger.log_param('parameters', params)
        neptune_logger.log_param('train_run_parameters', train_run_params)
        neptune_logger.add_tags([train_run_params['model_name'], 'test', run_name])

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
            dataloader=dataloader,
            pad_idx=word_to_ix[PAD],
            device=device,
            ix_to_tag=ix_to_tag,
            ix_to_word=ix_to_word,
            unk_idx=word_to_ix[UNK],
            labels=labels
        )
        
        neptune_logger.log_param('metrics', eval_res['metrics'])
        neptune_logger.log_json('results/unk_foreach_tag', eval_res['unk_foreach_tag'])
        neptune_logger.log_txt('metrics/flat_classification_report', eval_res['flat_classification_report'])
        neptune_logger.log_figure('results/diagram', eval_res['fig'])
        neptune_logger.log_colorized_table('results/predicted', eval_res['df_predicted'], eval_res['matched_indices'], eval_res['false_positive_indices'], eval_res['false_negative_indices'])
        neptune_logger.log_table('results/actual', eval_res['df_actual'])

        # TODO: unknown words appear as UNK in resulted tables
