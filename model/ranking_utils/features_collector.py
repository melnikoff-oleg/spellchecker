from typing import Dict, Callable
from model.ranking_utils.features import *


class FeaturesCollector:
    def __init__(self, features_names: List[str]):
        self._all_features: Dict[str, Callable[[], BaseFeature]] = {
            "levenshtein": lambda: LevenshteinFeature(),
            "bart_prob": lambda: BartProbFeature(),
        }
        self._features = {fname: self._all_features[fname]() for fname in features_names}
        self._features_names = features_names

    def collect(self, spelled_words: List[SpelledWord], candidates: List[List[str]]) -> List[List[List[float]]]:
        if not candidates:
            return []
        all_features: List[List[List[float]]] = [[[] for _ in candidates[idx]] for idx, _ in enumerate(spelled_words)]
        if not candidates:
            return all_features
        for jdx, feature_name in enumerate(self._features_names):
            feature_values = self._features[feature_name].compute_candidates(spelled_words, candidates)
            for idx, spelled_word in enumerate(spelled_words):
                for kdx, value in enumerate(feature_values[idx]):
                    all_features[idx][kdx].append(value)
        return all_features

    @property
    def features_names(self):
        return self._features_names
