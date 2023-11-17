import os

from ...constants.artifacts import PATHS
from ..data_master.data_generator import DataGenerator
from ...constants.nn import NUM_WORDS, PAD, UNK


def get_sents_from_dataset(dataset: str, case_sensitive: bool = False) -> list[tuple[list[str], list[str]]]:
    with open(get_dataset_path(dataset), encoding='utf-8') as file:
        sents = DataGenerator.generate_sents2(file)
    
    if not case_sensitive:
        for index, (words, labels) in enumerate(sents):
            sents[index] = ([word.lower() for word in words], labels)
    
    return sents


def get_dataset_path(dataset: str) -> str:
    return os.path.join(PATHS['DATASETS_PATH'], f'{dataset}.txt')


def generate_tag_to_ix_from_sents(sents: list[tuple[list[str], list[str]]]) -> dict[str, int]:
    labels_set = set()

    for _, labels in sents:
        labels_set.update(labels)

    return {key: index for index, key in enumerate(sorted(labels_set))}


def generate_ix_to_tag(tag_to_ix: dict[str, int]):
    return {index: key for key, index in tag_to_ix.items()}


def generate_labels(tag_to_ix: dict[str, int]):
    lst = [None] * len(tag_to_ix)
    for word, index in tag_to_ix.items():
        lst[index] = word
    return lst


def generate_word_to_ix(sents: list[tuple[list[str], list[str]]], num2words: bool = True, case_sensitive: bool = False):
    words_set = set()

    if case_sensitive:
        for words, _ in sents:
            words_set.update(words)
    else:
        for words, _ in sents:
            words_set.update([word.lower() for word in words])
    
    if num2words:
        if case_sensitive:
            words_set.update(NUM_WORDS)
        else:
            words_set.update([word.lower() for word in NUM_WORDS])
    
    words_set.update([PAD, UNK])
    
    return {word: index for index, word in enumerate(sorted(words_set))}
