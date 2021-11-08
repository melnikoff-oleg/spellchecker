import os
from typing import List, Dict, Callable

from grazie.common.main.file import read_lines, read_json
from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.bart_prob import BartProbFeature
from grazie.spell.main.model.features.base import BaseFeature
from grazie.spell.main.model.features.bert import BertProbFeature
from grazie.spell.main.model.features.edit_dist import LevenshteinFeature, JaroWinklerFeature
from grazie.spell.main.model.features.freq import FreqFeature, SqrtFreqFeature, LogFreqFeature
from grazie.spell.main.model.features.phonetic import MetaphonePhoneticFeature, SoundexPhoneticFeature
from grazie.spell.main.model.features.suffix_prob import SuffixProbFeature


class FeaturesCollector:
    def __init__(self, features_names: List[str], freqs: Dict[str, float]):
        self._all_features: Dict[str, Callable[[], BaseFeature]] = {
            "levenshtein": lambda: LevenshteinFeature(),
            "jaro_winkler": lambda: JaroWinklerFeature(),

            "freq": lambda: FreqFeature(freqs),
            "log_freq": lambda: LogFreqFeature(freqs),
            "sqrt_freq": lambda: SqrtFreqFeature(freqs),

            "soundex": lambda: SoundexPhoneticFeature(),
            "metaphone": lambda: MetaphonePhoneticFeature(),

            "bert_prob": lambda: BertProbFeature(),
            "suffix_prob": lambda: SuffixProbFeature(),
            "bart_prob": lambda: BartProbFeature()
        }

        self._features = {fname: self._all_features[fname]() for fname in features_names}
        self._features_names = features_names

    @classmethod
    def load(cls, path: str):
        config = read_json(os.path.join(path, "config.json"))
        freqs = FeaturesCollector.load_freqs(os.path.join(path, "freqs.dic"))
        return FeaturesCollector(config["features_names"], freqs)

    def collect(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[List[float]]:
        if not candidates:
            return []
        all_features: List[List[float]] = [[] for _ in candidates]
        for feature_name in self._features_names:
            features = self._features[feature_name].compute_candidates(text, spelled_word, candidates)
            for i, f_value in enumerate(features):
                all_features[i].append(f_value)
        return all_features

    @property
    def features_names(self):
        return self._features_names

    @classmethod
    def load_freqs(cls, freqs_path: str) -> Dict[str, float]:
        freqs: Dict[str, float] = {}
        for line in read_lines(freqs_path):
            word, freq = line.split('\t')
            freqs[word] = float(freq)
        return freqs
