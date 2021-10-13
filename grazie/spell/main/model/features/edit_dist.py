from typing import List

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
