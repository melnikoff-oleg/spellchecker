from typing import List, Any

import nltk

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class LevenshteinFeature(BaseFeature):
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = []
        for candidate in candidates:
            dist = nltk.edit_distance(spelled_word.word, candidate, transpositions=True)
            res.append(dist)
        return res


class JaroWinklerFeature(BaseFeature):
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = []
        for candidate in candidates:
            dist = nltk.metrics.distance.jaro_winkler_similarity(spelled_word.word, candidate)
            res.append(dist)
        return res


def levenshtein_dist(frames1: List[Any], weights1: List[float], frames2: List[Any], weights2: List[float]) -> float:
    matrix = [[0.0 for _ in range(len(frames1) + 1)] for _ in range(len(frames2) + 1)]

    prev_column = matrix[0]

    for i in range(len(frames1)):
        prev_column[i + 1] = prev_column[i] + weights1[i]

    if len(frames1) == 0 or len(frames2) == 0:
        return 0.0

    curr_column = matrix[1]

    for i2 in range(len(frames2)):

        frame2 = frames2[i2]
        weight2 = weights2[i2]

        curr_column[0] = prev_column[0] + weight2

        for i1 in range(len(frames1)):

            frame1 = frames1[i1]
            weight1 = weights1[i1]

            if frame1 == frame2:
                curr_column[i1 + 1] = prev_column[i1]
            else:
                change = weight1 + weight2 + prev_column[i1] # здесь заменить на вес от двух букв вместо суммы
                remove = weight2 + prev_column[i1 + 1]
                insert = weight1 + curr_column[i1]

                curr_column[i1 + 1] = min(change, remove, insert)

        if i2 != len(frames2) - 1:
            prev_column = curr_column
            curr_column = matrix[i2 + 2]

