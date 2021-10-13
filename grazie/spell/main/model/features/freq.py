from math import log, sqrt
from typing import Dict, List

from grazie.common.main.file import read_lines
from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class FreqFeature(BaseFeature):
    def __init__(self, freqs: Dict[str, float]):
        self._freqs: Dict[str, float] = freqs
        self._eps = 0.001

    @classmethod
    def load(cls, freqs_path: str) -> 'FreqFeature':
        freqs: Dict[str, float] = {}
        for line in read_lines(freqs_path):
            word, freq = line.split('\t')
            freqs[word] = int(freq)
        return FreqFeature(freqs)

    def transform(self, freq: float) -> float:
        return freq

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        return [self.transform(self._freqs.get(candidate, self._eps)) for candidate in candidates]


class LogFreqFeature(FreqFeature):
    def transform(self, freq: float) -> float:
        return log(freq)


class SqrtFreqFeature(FreqFeature):
    def transform(self, freq: float) -> float:
        return sqrt(freq)
