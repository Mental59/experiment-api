from .generator import get_labels_from_sents


def sents_has_unknown_labels(sents: list[tuple[list[str], list[str]]], tag_to_ix: dict[str, int]) -> bool:
    sents_labels = get_labels_from_sents(sents)

    for tag in tag_to_ix:
        if tag not in sents_labels:
            return True
    
    return False
