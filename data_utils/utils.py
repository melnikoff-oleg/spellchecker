from argparse import ArgumentParser
from typing import Tuple, List

from data_utils.base import SpelledText, Spell


def default_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--texts_path", type=str, required=True, help="File with correct texts")
    parser.add_argument("--size", type=int, required=False, default=50, help="Number of texts from file")
    return parser


def get_texts_from_file(path: str, char_based: bool = False):
    texts = []
    with open(path) as f:
        for line in f:
            text = line[:-1]
            if char_based:
                text.replace(' ', '_')
            texts.append(text)
    return texts


def read_data(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i[:-1], j[:-1]]))
    return data


def read_data_char_based(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i[:-1], j[:-1]]))
    for ind, i in enumerate(data):
        data[ind] = (i[0].replace(' ', '_'), i[1].replace(' ', '_'))
    return data


def get_parallel_texts_from_files(file1_path: str, file2_path: str, char_based: bool = False):
    res = []
    for text1, text2 in zip(get_texts_from_file(file1_path, char_based), get_texts_from_file(file2_path, char_based)):
        res.append([text1, text2])
    return res


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
