import torch
import neptune
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn import metrics

from ....services.nn.utils import CustomDataset, train, get_model_mean_confidence, flatten_list
from ....services.nn.bilstm_crf import BiLSTM_CRF
from ....services.data_master import count_unk_foreach_tag, DataAnalyzer
from ....models.ml.model_enum import ModelEnum
from ....services.dataset_processor import generator as dataset_generator
from ....services.experiment import logger as experiment_logger, setupper as experiment_setupper
from ....constants.nn import PAD, UNK


def run(
    project: str,
    run_name: str,
    api_token: str,
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
):
    model = str(ModelEnum.BiLSTM_CRF)
    device = experiment_setupper.get_torch_device()

    run = neptune.init_run(
        project=project,
        api_token=api_token,
        capture_stderr=True,
        capture_stdout=True,
        capture_traceback=True,
        capture_hardware_metrics=True,
        dependencies='infer'
    )
    
    try:
        params = {
            'model_name': model,
            'dataset': dataset,
            'device': device,
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
        run['parameters'] = params
        run['sys/tags'].add([model, 'train', run_name])

        sents = dataset_generator.get_sents_from_dataset(dataset, case_sensitive=case_sensitive)
        experiment_logger.log_by_path_neptune(run, 'data/dataset', dataset_generator.get_dataset_path(dataset))

        tag_to_ix = dataset_generator.generate_tag_to_ix_from_sents(sents)
        ix_to_tag = dataset_generator.generate_ix_to_key(tag_to_ix)
        experiment_logger.log_json_neptune(run, tag_to_ix, 'data/tag_to_ix')

        word_to_ix = dataset_generator.generate_word_to_ix(sents, num2words=num2words, case_sensitive=case_sensitive)
        ix_to_word = dataset_generator.generate_ix_to_key(word_to_ix)
        experiment_logger.log_json_neptune(run, word_to_ix, 'data/word_to_ix')

        train_data, val_data = train_test_split(sents, test_size=test_size)
        train_dataset = CustomDataset(train_data, tag_to_ix, word_to_ix, convert_nums2words=num2words)
        val_dataset = CustomDataset(val_data, tag_to_ix, word_to_ix, convert_nums2words=num2words)
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        }

        vocab_size = len(word_to_ix)
        num_tags = len(tag_to_ix)
        labels=dataset_generator.generate_labels(tag_to_ix)
        model = BiLSTM_CRF(vocab_size=vocab_size, num_tags=num_tags, embedding_dim=embedding_dim, hidden_dim=hidden_dim, padding_idx=word_to_ix[PAD]).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

        model, _ = train(
            model=model,
            optimizer=optimizer,
            dataloaders=dataloaders,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler,
            neptune_run=run,
            verbose=False
        )

        y_val_pred = []
        y_val_true = []
        X_val_indices = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch, _ in dataloaders['val']:
                X_val_indices.extend([torch.tensor([word_index for word_index in sent if word_index != word_to_ix[PAD]]) for sent in x_batch])

                x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
                y_batch_pred = model(x_batch, mask_batch)

                y_val_pred.extend(y_batch_pred)
                y_val_true.extend([[tag.item() for tag in tags if tag >= 0] for tags in y_batch])
        y_val_pred = [[ix_to_tag[tag] for tag in sentence] for sentence in y_val_pred]
        y_val_true = [[ix_to_tag[tag] for tag in sentence] for sentence in y_val_true]

        unk_foreach_tag = count_unk_foreach_tag(X_val_indices, y_val_true, labels, word_to_ix[UNK])
        experiment_logger.log_json_neptune(run, unk_foreach_tag, 'results/unk_foreach_tag')

        conf = get_model_mean_confidence(model, X_val_indices, device)
        run['metrics/confidence'] = conf

        y_val_true_flat = flatten_list(y_val_true)
        y_val_pred_flat = flatten_list(y_val_pred)

        run['metrics/f1_weighted'] = metrics.f1_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels)
        run['metrics/precision_weighted'] = metrics.precision_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels)
        run['metrics/recall_weighted'] = metrics.recall_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels)
        run['metrics/accuracy'] = metrics.accuracy_score(y_val_true_flat, y_val_pred_flat)

        flat_class_report = metrics.classification_report(y_val_true_flat, y_val_pred_flat, labels=labels, digits=3)
        experiment_logger.log_txt_neptune(run, flat_class_report, 'metrics/flat_classification_report')

        df_predicted, df_actual, fig = DataAnalyzer.analyze(
            X=[[ix_to_word[word_idx.item()] for word_idx in word_indices] for word_indices in X_val_indices],
            y_true=y_val_true,
            y_pred=y_val_pred,
            keys=labels
        )
        experiment_logger.log_figure_neptune(run, 'results/diagram', fig)
        experiment_logger.log_table_neptune(run, 'results/predicted', df_predicted)
        experiment_logger.log_table_neptune(run, 'results/actual', df_actual)

    finally:
        run.stop()
