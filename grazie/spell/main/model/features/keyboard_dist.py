from typing import List

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class QwertyFeature(BaseFeature):

    keyboard = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    ]

    def index_list_2d(list_2d: List[List[object]], element: object) -> tuple[int, int]:
        for row_index, row in enumerate(list_2d):
            try:
                column_index = row.index(element)
            except ValueError:
                continue
            return row_index, column_index
        return 2, 5

    def qwerty_index_of_char(char: str):
        return QwertyFeature.index_list_2d(QwertyFeature.keyboard, char)

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = []
        for candidate in candidates:
            padded_init_word = spelled_word.word
            padded_candidate = candidate
            if len(padded_init_word) < len(padded_candidate):
                padded_init_word += '#' * (len(padded_candidate) - len(padded_init_word))
            else:
                padded_candidate += '#' * (len(padded_init_word) - len(padded_candidate))

            dist = 0
            for char_index, char_init_word in enumerate(padded_init_word):
                char_candidate = padded_candidate[char_index]
                if char_init_word != char_candidate:
                    x_candidate, y_candidate = QwertyFeature.qwerty_index_of_char(char_candidate)
                    x_init_word, y_init_word = QwertyFeature.qwerty_index_of_char(char_init_word)
                    dist = abs(x_candidate - x_init_word) + abs(y_candidate - y_init_word)
                    break
            res.append(dist)
        return res
