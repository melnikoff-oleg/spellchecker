from abc import ABC, abstractmethod
from typing import List

from grazie.common.main.ranking.ranker import Ranker
from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.features_collector import FeaturesCollector


class SpellRanker(ABC):
    @abstractmethod
    def rank(self, text: str, spelled_word: SpelledWord, candidates: List[str], **kwargs) -> List[float]:
        raise NotImplementedError


class RandomSpellRanker(SpellRanker):
    def rank(self, text: str, spelled_word: SpelledWord, candidates: List[str], **kwargs) -> List[float]:
        return [0.0 for _ in candidates]


class FeaturesSpellRanker(SpellRanker):
    def __init__(self, features_collector: FeaturesCollector, ranking_model: Ranker):
        self._features_collector = features_collector
        self._ranking_model = ranking_model

    def rank(self, text: str, spelled_word: SpelledWord, candidates: List[str], **kwargs) -> List[float]:
        all_features = self._features_collector.collect(text, spelled_word, candidates)
        return self._ranking_model.predict(all_features)
