from abc import abstractmethod, ABC
from typing import List, Optional

from dataclasses import dataclass

from grazie.spell.main.model.candidator import BaseCandidator
from grazie.spell.main.model.detector import BaseDetector
from grazie.spell.main.model.ranker import SpellRanker


@dataclass
class SpellCheckVariant:
    substitution: str
    score: float
    absolutely_best: bool = False


@dataclass
class SpellCheckResult:
    start: int
    finish: int
    variants: List[SpellCheckVariant]


class SpellCheckModelBase(ABC):
    def __init__(self, max_count: int = 5):
        self.max_count = max_count

    # тут что ли какой-то другой max_count, не из self?
    @abstractmethod
    def check(self, text: str, max_count: Optional[int] = None, round_digits: int = None, **kwargs) -> List[SpellCheckResult]:
        raise NotImplementedError


class SpellCheckModel(SpellCheckModelBase):
    def __init__(self, detector: BaseDetector, candidator: BaseCandidator, ranker: SpellRanker, max_count: Optional[int] = None):
        super().__init__(max_count)
        self.detector = detector
        self.candidator = candidator
        self.ranker = ranker

    def check(self, text: str, max_count: Optional[int] = None, round_digits: int = None, **kwargs) -> List[SpellCheckResult]:
        # что значит конструкция int or int?
        max_count = max_count or self.max_count
        round_digits = round_digits or 100

        spelled_words = self.detector.detect(text, **kwargs)
        all_candidates = self.candidator.get_candidates(text, spelled_words, **kwargs)

        scored_candidates = []
        for i, (spelled_word, candidates) in enumerate(zip(spelled_words, all_candidates)):
            scores = self.ranker.rank(text, spelled_word, candidates, **kwargs)
            variants = [SpellCheckVariant(candidate, round(score, round_digits), False) for score, candidate in sorted(zip(scores, candidates), reverse=True)]

            spell_check_result = SpellCheckResult(spelled_word.interval[0], spelled_word.interval[1], variants[:max_count])
            scored_candidates.append(spell_check_result)

        assert len(spelled_words) == len(scored_candidates)

        return scored_candidates


# class JemSpellModel(SpellCheckModelBase):
#     def __init__(self):
#         self._jsp = jamspell.TSpellCorrector()
#         assert self._jsp.LoadLangModel('en.bin')
