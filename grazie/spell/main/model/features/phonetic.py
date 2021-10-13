from abc import ABC
from typing import List

from fonetika.distance import PhoneticsInnerLanguageDistance
from fonetika.metaphone import Metaphone
from fonetika.soundex import EnglishSoundex

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class PhoneticFeature(BaseFeature, ABC):
    # alternatives - pyphonetics, phonetics
    def __init__(self, algo):
        self._algo = algo
        self._distance = PhoneticsInnerLanguageDistance(self._algo, metric_name='levenstein')

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        res = []
        for candidate in candidates:
            try:
                res.append(self._distance.distance(spelled_word.word, candidate))
            except IndexError:
                res.append(len(candidate))
        return res


class SoundexPhoneticFeature(PhoneticFeature):
    def __init__(self):
        super().__init__(EnglishSoundex())


class MetaphonePhoneticFeature(PhoneticFeature):
    def __init__(self):
        super().__init__(Metaphone())
