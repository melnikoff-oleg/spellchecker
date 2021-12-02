from typing import List

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature
from grazie.spell.main.model.features.fill_text_prob import FillTextProbComputer
from grazie.spell.test.feature_tester import test_feature


class BartProbFeature(BaseFeature):
    def __init__(self, name: str = "facebook/bart-base"):
        self._model = FillTextProbComputer(name)

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        request = (spelled_word.text, spelled_word.interval[0], spelled_word.interval[1])
        scores = self._model.log_probs([request], [candidates])[0]
        return scores


if __name__ == '__main__':
    test_feature(BartProbFeature())
