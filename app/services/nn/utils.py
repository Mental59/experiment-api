import io
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics

from .bilstm_crf import BiLSTM_CRF
from .custom_dataset import CustomDataset
from ..data_master import count_unk_foreach_tag, DataAnalyzer
from ...services.experiment.logger import NeptuneLogger
from ...models.ml.eval_res import MetricsEvaluateRes, EvaluateRes
from ...constants.nn import UNKNOWN_TAG


def train(
        model,
        optimizer,
        dataloaders,
        device,
        num_epochs,
        neptune_logger: Optional[NeptuneLogger]=None,
        scheduler=None,
        verbose=True):
    losses = {'train': [], 'val': []}
    best_loss = None
    model_buffer = io.BytesIO()

    for epoch in tqdm(range(1, num_epochs + 1)):
        losses_per_epoch = {'train': 0.0, 'val': 0.0}

        if neptune_logger is not None:
            neptune_logger.log_param('epoch', epoch)

        model.train()
        for x_batch, y_batch, mask_batch, custom_features in dataloaders['train']:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            custom_features = custom_features.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch, custom_features)
            loss.backward()
            optimizer.step()

            if neptune_logger is not None:
                neptune_logger.append_param('train/batch/loss', loss.item())

            losses_per_epoch['train'] += loss.item()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch, custom_features in dataloaders['val']:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                custom_features = custom_features.to(device)
                loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch, custom_features)

                if neptune_logger is not None:
                    neptune_logger.append_param('val/batch/loss', loss.item())

                losses_per_epoch['val'] += loss.item()

        for mode in ['train', 'val']:
            losses_per_epoch[mode] /= len(dataloaders[mode])
            losses[mode].append(losses_per_epoch[mode])

        if best_loss is None or best_loss > losses_per_epoch['val']:
            best_loss = losses_per_epoch['val']
            model_buffer.seek(0)
            torch.save(model.state_dict(), model_buffer)

        if scheduler is not None:
            scheduler.step(losses_per_epoch['val'])

        if verbose:
            print(
                'Epoch: {}'.format(epoch),
                'train_loss: {}'.format(losses_per_epoch['train']),
                'val_loss: {}'.format(losses_per_epoch['val']),
                sep=', '
            )

    if neptune_logger is not None:
        model_buffer.seek(0)
        neptune_logger.log_binary('model_checkpoints/best_model', model_buffer.read(), 'pth')

        model_buffer.seek(0)
        model.load_state_dict(torch.load(model_buffer))

    return model, losses


def evaluate_model(
        model: BiLSTM_CRF,
        dataset: CustomDataset,
        dataloader: DataLoader,
        pad_idx: int,
        unk_idx: int,
        device: str,
        ix_to_tag: dict[int, str],
        labels: list[str]
    ):
    y_val_pred = []
    y_val_true = []
    X_val_indices = []

    with torch.no_grad():
        for x_batch, y_batch, mask_batch, _ in dataloader:
            X_val_indices.extend([torch.tensor([word_index for word_index in sent if word_index != pad_idx]) for sent in x_batch])

            x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
            y_batch_pred = model(x_batch, mask_batch)

            y_val_pred.extend(y_batch_pred)
            y_val_true.extend([[tag.item() for tag in tags if tag >= 0] for tags in y_batch])

    y_val_pred = [[ix_to_tag.get(tag, UNKNOWN_TAG) for tag in sentence] for sentence in y_val_pred]
    y_val_true = [[ix_to_tag.get(tag, UNKNOWN_TAG) for tag in sentence] for sentence in y_val_true]

    unk_foreach_tag = count_unk_foreach_tag(X_val_indices, y_val_true, labels, unk_idx)
    conf = get_model_mean_confidence(model, X_val_indices, device)

    y_val_true_flat = flatten_list(y_val_true)
    y_val_pred_flat = flatten_list(y_val_pred)

    m = MetricsEvaluateRes(
        f1_weighted=metrics.f1_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels),
        precision_weighted=metrics.precision_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels),
        recall_weighted=metrics.recall_score(y_val_true_flat, y_val_pred_flat, average='weighted', labels=labels),
        accuracy=metrics.accuracy_score(y_val_true_flat, y_val_pred_flat),
        confidence=conf
    )

    flat_classification_report = metrics.classification_report(y_val_true_flat, y_val_pred_flat, labels=labels, digits=3)

    df_predicted, df_actual, fig, matched_indices, false_positive_indices, false_negative_indices = DataAnalyzer.analyze(
        X=[sentence for sentence, _ in dataset.raw_data()],
        y_true=y_val_true,
        y_pred=y_val_pred,
        keys=labels
    )

    return EvaluateRes(
        unk_foreach_tag=unk_foreach_tag,
        metrics=m,
        flat_classification_report=flat_classification_report,
        df_predicted=df_predicted,
        df_actual=df_actual,
        fig=fig,
        matched_indices=matched_indices,
        false_positive_indices=false_positive_indices,
        false_negative_indices=false_negative_indices
    )


def plot_losses(losses, figsize=(12, 8), savepath: str = None, show=True):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=figsize)
    for mode in ['train', 'val']:
        plt.plot(losses[mode], label=mode)
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()


def generate_tag_to_ix(keys: list):
    tag_to_ix = {}
    i = 0
    for key in keys:
        tag_to_ix[key] = i
        i += 1
    return tag_to_ix


def get_model_confidence(
        model: nn.Module, X_test: List[torch.Tensor], device, test_dataset: CustomDataset = None) -> List[float]:
    """Computes model's confidence for each sentence in X_test"""
    confs = []
    with torch.no_grad():
        for index, sentence in enumerate(X_test):
            sentence = sentence.unsqueeze(0).to(device)

            f = None
            if test_dataset is not None:
                _, _, _, custom_features = test_dataset[index]
                if custom_features is not None:
                    f = custom_features[:sentence.size(1), ...].unsqueeze(0).to(device)

            best_tag_sequence = model(sentence, custom_features=f)
            confidence = torch.exp(
                -model.neg_log_likelihood(
                    sentence,
                    torch.tensor(best_tag_sequence, device=device),
                    custom_features=f
                )
            )
            confs.append(confidence.item())

    return confs

def get_model_mean_confidence(
        model: nn.Module,
        X_test: List[torch.Tensor],
        device,
        test_dataset: CustomDataset = None) -> float:
    """Computes model's confidence for each sentence in X_test"""
    conf = 0
    with torch.no_grad():
        for index, sentence in tqdm(enumerate(X_test), desc='get_model_mean_confidence'):
            sentence = sentence.unsqueeze(0).to(device)

            f = None
            if test_dataset is not None:
                _, _, _, custom_features = test_dataset[index]
                if custom_features is not None:
                    f = custom_features[:sentence.size(1), ...].unsqueeze(0).to(device)

            best_tag_sequence = model(sentence, custom_features=f)
            confidence = torch.exp(
                -model.neg_log_likelihood(
                    sentence,
                    torch.tensor(best_tag_sequence, device=device),
                    custom_features=f
                )
            )
            conf += confidence.item()

    return conf / len(X_test)


def flatten_list(lst: list):
    return [item for row in lst for item in row]
