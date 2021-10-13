from abc import abstractmethod, ABC
from typing import List, Dict, Set

import nltk
from hunspell import Hunspell
from nltk.corpus import words as nltk_words

from grazie.spell.main.model.base import SpelledWord


class BaseCandidator(ABC):
    @abstractmethod
    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        raise NotImplementedError


class IdealCandidator(BaseCandidator):
    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        true_spells = kwargs["true_spells"]
        candidates = []
        for spelled_word in spelled_words:
            cands = []
            for spell in true_spells:
                if spelled_word.word == spell.spelled:
                    cands.append(spell.correct)
                    break
            candidates.append(cands)
        return candidates


class AggregatedCandidator(BaseCandidator):
    def __init__(self, candidators: List[BaseCandidator]):
        self._candidators = candidators

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for candidator in self._candidators:
            candidates = candidator.get_candidates(text, spelled_words, **kwargs)
            for i, candidate in enumerate(candidates):
                all_candidates[i].extend(candidate)

        for i in range(len(all_candidates)):
            all_candidates[i] = list(set(all_candidates[i]))

        return all_candidates


class LevenshteinCandidator(BaseCandidator):
    def __init__(self, max_err: int = 2, index_prefix_len: int = 0):
        self.require_nltk()

        self._dict: Dict[str, Set[str]] = {}
        self._max_err = max_err
        self._prefix_len = index_prefix_len

        # lines = read_lines(dict_path)
        # words = filter(lambda x: x.strip(), map(lambda x: x.split('\t')[0].split(',')[0], lines))
        words = set(word.lower() for word in nltk_words.words())

        for word in words:
            key = word[:self._prefix_len]
            if key not in self._dict:
                self._dict[key] = set()
            self._dict[key].add(word)

    @staticmethod
    def require_nltk():
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        candidates: List[List[str]] = [[] for _ in spelled_words]
        for i, spelled_word in enumerate(spelled_words):
            key = spelled_word.word[:self._prefix_len]
            if key not in self._dict:
                continue
            for candidate_word in self._dict[key]:
                edit_distance = nltk.edit_distance(candidate_word, spelled_word.word, transpositions=True)
                if edit_distance <= self._max_err:
                    candidates[i].append(candidate_word)
        return candidates


class HunspellCandidator(BaseCandidator):
    def __init__(self):
        self._hunspell = Hunspell()

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for i, spelled_word in enumerate(spelled_words):
            candidates = self._hunspell.suggest(spelled_word.word)
            all_candidates[i] = list(candidates)
        return all_candidates
