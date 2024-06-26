from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from app.core.decorators import single_experiment_run

from ....services.nn.utils import CustomDataset, train, evaluate_model
from ....services.nn.bilstm_crf import LSTM_CRF
from ....models.ml.model_enum import ModelEnum
from ....services.dataset_processor import generator as dataset_generator
from ....services.experiment import setupper as experiment_setupper
from ....services.experiment.tracker.utils import get_experiment_tracker
from ....constants.nn import PAD, UNK
from ....models.ml.experiment_tracker_enum import ExperimentTrackerEnum
from ....constants.save_keys import *


@single_experiment_run
def run(
    project: str,
    run_name: str,
    dataset: str,
    embedding_dim: int,
    hidden_dim: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    scheduler_factor: float,
    scheduler_patience: int,
    weight_decay: float,
    case_sensitive: bool,
    test_size: float,
    num2words: bool,
    experiment_tracker_type: str,
    **kwargs
):
    with get_experiment_tracker(experiment_tracker_type, project=project, run_name=run_name, **kwargs) as experiment_tracker:
        model = str(ModelEnum.LSTM_CRF)
        device = experiment_setupper.get_torch_device()

        params = {
            'model_name': model,
            'dataset': dataset,
            'device': str(device),
            'batch_size': batch_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'scheduler_factor': scheduler_factor,
            'scheduler_patience': scheduler_patience,
            'case_sensitive': case_sensitive,
            'weight_decay': weight_decay,
            'test_size': test_size,
            'num2words': num2words,
        }
        experiment_tracker.log_params(PARAMETERS_SAVE_KEY, params)
        experiment_tracker.add_tags(dict(model_name=model, mode='train', run_name=run_name))

        sents = dataset_generator.get_sents_from_dataset(dataset, case_sensitive=case_sensitive)
        experiment_tracker.log_dataset(DATASET_SAVE_KEY, dataset_generator.get_dataset_path(dataset))

        tag_to_ix = dataset_generator.generate_tag_to_ix_from_sents(sents)
        ix_to_tag = dataset_generator.generate_ix_to_key(tag_to_ix)
        experiment_tracker.log_json(TAG_TO_IX_SAVE_KEY, tag_to_ix)

        word_to_ix = dataset_generator.generate_word_to_ix(sents, num2words=num2words, case_sensitive=case_sensitive)
        experiment_tracker.log_json(WORD_TO_IX_SAVE_KEY, word_to_ix)

        train_data, val_data = train_test_split(sents, test_size=test_size)
        train_dataset = CustomDataset(train_data, tag_to_ix, word_to_ix, convert_nums2words=num2words)
        val_dataset = CustomDataset(val_data, tag_to_ix, word_to_ix, convert_nums2words=num2words)
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        }

        vocab_size = len(word_to_ix)
        num_tags = len(tag_to_ix)
        labels = dataset_generator.generate_labels(tag_to_ix)
        model = LSTM_CRF(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            padding_idx=word_to_ix[PAD],
        ).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

        model, _ = train(
            model=model,
            optimizer=optimizer,
            dataloaders=dataloaders,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler,
            experiment_logger=experiment_tracker,
            verbose=False
        )

        model.eval()
        eval_res = evaluate_model(
            model=model,
            dataloader=dataloaders['val'],
            dataset=val_dataset,
            pad_idx=word_to_ix[PAD],
            device=device,
            ix_to_tag=ix_to_tag,
            unk_idx=word_to_ix[UNK],
            labels=labels
        )

        experiment_tracker.log_metrics(eval_res.metrics.model_dump())
        experiment_tracker.log_json(UNK_FOREACH_TAG_SAVE_KEY, eval_res.unk_foreach_tag)
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

        return experiment_tracker.get_run_result(), eval_res.metrics, params
