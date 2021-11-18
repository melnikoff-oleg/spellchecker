from typing import List

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class CandidateLengthDiff(BaseFeature):
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = []
        for candidate in candidates:
            res.append(abs(len(spelled_word.word) - len(candidate)))
        return res


class InitWordLength(BaseFeature):
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = [len(spelled_word.word) for candidate in candidates]
        return res

# добавить разницу в длинах
