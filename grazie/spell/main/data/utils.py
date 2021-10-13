from argparse import ArgumentParser
from typing import Tuple, List

from grazie.spell.main.data.base import SpelledText
# from grazie.spell.main.data.generate_synthetic import generate_spelling_synthetic_data


def default_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--texts_path", type=str, required=True, help="File with correct texts")
    parser.add_argument("--size", type=int, required=False, default=50, help="Number of texts from file")
    return parser


def get_test_data(texts_path: str, size: int = None) -> Tuple[List[SpelledText], List[SpelledText]]:
    texts = []
    with open(texts_path) as f:
        for i, line in f:
            texts.append(line)
            if len(texts) >= size:
                break
    # spelling_data = generate_spelling_synthetic_data(texts, 1.0)
    # sorry :(
    spelling_data = []
    train_data, test_data = spelling_data[:int(len(spelling_data) * 0.7)], spelling_data[int(len(spelling_data) * 0.7):]
    return train_data, test_data
