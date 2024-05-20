from typing import Literal
import pandas as pd
from datasets import Dataset
from sklearn import metrics

from app.models.ml.eval_res import EvaluateRes, MetricsEvaluateRes
from app.services.data_master.data_analyzer import DataAnalyzer
from app.services.data_master.utils import count_unk_foreach_tag
from app.services.nn.utils import flatten_list


NerPipelineOutput = list[list[dict[Literal['entity', 'score', 'index', 'word', 'start', 'end'], str | int | float]]]


def get_token_dataset(
    sents,
    task = 'ner'
) -> Dataset:
    df = pd.DataFrame(sents, columns=('tokens', f'{task}_tags'))
    df['whole_string'] = df['tokens'].apply(' '.join)
    return Dataset.from_pandas(df)


def get_corpus_generator(dataset: Dataset):
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx:start_idx + 1000]
        yield samples["whole_string"]


def tokenize_and_align_labels(
    dataset,
    tokenizer,
    tag_to_ix,
    task = 'ner',
    label_all_tokens = True
):
    tokenized_inputs = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(dataset[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(tag_to_ix[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(tag_to_ix[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def get_extended_y_val_true(sents, outputs):
    sent_ranges = []
    for sentence, _ in sents:
        lst = []
        last_stop = -1
        for word in sentence:
            start = last_stop + 1
            stop = len(word) + start
            lst.append(range(start, stop))
            last_stop = stop
        sent_ranges.append(lst)
    
    y_val_true = []

    for output, (sentence, tags), ranges in zip(outputs, sents, sent_ranges):
        lst = []
        for val in output:
            found_range_index = None
            for index, r in enumerate(ranges):
                if val['start'] >= r.start and val['end'] <= r.stop:
                    found_range_index = index
                    break
            lst.append(tags[found_range_index])
        y_val_true.append(lst)
    
    return y_val_true


def evaluate_transformer_pipeline(pipeline, sents, batch_size: int | None = None, num_workers: int | None = None):
    whole_strings = [' '.join(sentence) for sentence, _ in sents]

    outputs: NerPipelineOutput = pipeline(whole_strings, batch_size=batch_size, num_workers=num_workers)
    labels = set()
    for _, tags in sents:
        labels.update(tags)
    labels = sorted(labels)

    # transformers tokenize words differently (for example word "oakridge" would be splitted into 2 tokens: "oak" and "ridge")
    # so we need to extend y_val_true tags
    y_val_true = get_extended_y_val_true(sents, outputs)
    
    y_val_pred = [[val['entity'] for val in output] for output in outputs]

    unk_foreach_tag = dict()
    conf = 0.0
    for output in outputs:
        conf += sum([val['score'] for val in output]) / len(output)
    conf /= len(outputs)

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
        X=[sentence for sentence, _ in sents],
        y_true=y_val_true,
        y_pred=y_val_pred,
        keys=labels
    )

    for output in outputs:
        for val in output:
            val['entity'] = str(val['entity'])
            val['score'] = float(val['score'])
            val['index'] = int(val['index'])
            val['word'] = str(val['word'])
            val['start'] = int(val['start'])
            val['end'] = int(val['end'])

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
    ), outputs
