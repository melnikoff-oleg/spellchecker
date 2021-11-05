import os
from typing import List, Dict, Callable

import json
# from grazie.common.main.file import read_lines, read_json
from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature
from grazie.spell.main.model.features.bert import BertProbFeature
from grazie.spell.main.model.features.edit_dist import LevenshteinFeature, JaroWinklerFeature
from grazie.spell.main.model.features.freq import FreqFeature, SqrtFreqFeature, LogFreqFeature
from grazie.spell.main.model.features.phonetic import MetaphonePhoneticFeature, SoundexPhoneticFeature
from grazie.spell.main.model.features.suffix_prob import SuffixProbFeature
from grazie.spell.main.model.features.keyboard_dist import QwertyFeature
from grazie.spell.main.model.features.count_candidates_less_edit_dist import CountCandidatesLessEditDistFeature
from grazie.spell.main.model.features.n_grams import BiGramsFeature, TriGramsFeature
from grazie.spell.main.model.features.words_length import CandidateLength, InitWordLength

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

            "keyboard_dist": lambda: QwertyFeature(),
            "cands_less_dist": lambda: CountCandidatesLessEditDistFeature(),

            "bigram_freq": lambda: BiGramsFeature(),
            "trigram_freq": lambda: TriGramsFeature(),

            "cand_length": lambda: CandidateLength(),
            "init_word_length": lambda: InitWordLength(),
        }

        self._features = {fname: self._all_features[fname]() for fname in features_names}
        self._features_names = features_names

    @classmethod
    def load(cls, path: str):
        config = json.load(os.path.join(path, "config.json"))
        # config = read_json(os.path.join(path, "config.json"))
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
        with open(freqs_path) as f:
            read_lines = f.readlines()
        # for line in read_lines(freqs_path):
        for line in read_lines:
            # word, freq = line.split('\t')
            word, freq = line.split(',')
            freqs[word] = float(freq)
        return freqs
