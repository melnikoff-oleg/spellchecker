from argparse import ArgumentParser
from typing import Tuple, List

from grazie.spell.main.data.base import SpelledText, Spell


def default_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--texts_path", type=str, required=True, help="File with correct texts")
    parser.add_argument("--size", type=int, required=False, default=50, help="Number of texts from file")
    return parser


def get_test_data(gt_texts_path: str, noisy_texts_path: str, size: int = None, train_part: float = 0.7) -> Tuple[List[SpelledText], List[SpelledText]]:

    with open(gt_texts_path) as f:
        gt_lines = f.readlines()
    with open(noisy_texts_path) as f:
        noise_lines = f.readlines()

    texts = []
    ind = 0
    for gt_, noise_ in zip(gt_lines, noise_lines):
        gt = gt_[:-1]
        noise = noise_[:-1]
        fict_noise = noise
        gt_words = gt.split()
        noise_words = noise.split()
        spells = []
        for gt_word, noise_word in zip(gt_words, noise_words):
            if gt_word != noise_word:
                start = fict_noise.find(noise_word)
                finish = start + len(noise_word)
                fict_noise = fict_noise[:start] + ''.join(['#' for ind in range(start, finish)]) + fict_noise[finish:]
                new_spell = Spell(spelled=noise_word, correct=gt_word, start=start)
                spells.append(new_spell)
        cur_spelled_text = SpelledText(noise, spells)
        texts.append(cur_spelled_text)
        ind += 1
        if ind == size:
            break

    train_data: List[SpelledText] = texts[:round(len(texts) * train_part)]
    test_data: List[SpelledText] = texts[round(len(texts) * train_part):]
    return train_data, test_data
