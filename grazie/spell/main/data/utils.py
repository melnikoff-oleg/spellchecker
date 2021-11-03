from argparse import ArgumentParser
from typing import Tuple, List

from grazie.spell.main.data.base import SpelledText, Spell
# from grazie.spell.main.data.generate_synthetic import generate_spelling_synthetic_data


def default_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--texts_path", type=str, required=True, help="File with correct texts")
    parser.add_argument("--size", type=int, required=False, default=50, help="Number of texts from file")
    return parser


def get_test_data(gt_texts_path: str, noisy_texts_path: str, size: int = None) -> Tuple[List[SpelledText], List[SpelledText]]:

    with open(gt_texts_path) as f:
        gt_lines = f.readlines()
    with open(noisy_texts_path) as f:
        noise_lines = f.readlines()

    texts = []
    ind = 0
    for gt, noise in zip(gt_lines, noise_lines):
        gt_words = gt.split()
        noise_words = noise.split()
        spells = []
        for gt_word, noise_word in zip(gt_words, noise_words):
            if gt_word != noise_word:
                new_spell = Spell(spelled=noise_word, correct=gt_word)
                spells.append(new_spell)
        cur_spelled_text = SpelledText(noise, spells)
        texts.append(cur_spelled_text)
        ind += 1
        if ind == size:
            break

    train_data: List[SpelledText] = texts[:round(len(texts) * 0.7)]
    test_data: List[SpelledText] = texts[round(len(texts) * 0.7):]
    return train_data, test_data