from abc import abstractmethod, ABC
from typing import List, Dict, Set

import nltk
import torch
from hunspell import Hunspell
from nltk.corpus import words as nltk_words

from grazie.spell.main.model.base import SpelledWord

import pkg_resources
from symspellpy import SymSpell, Verbosity

from grazie.spell.main.model.generating.swap_word_generator import SwapWordGenerator


class BaseCandidator(ABC):
    @abstractmethod
    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        raise NotImplementedError

# этот класс возвращает по одному gt кандидату на слово
class IdealCandidator(BaseCandidator):
    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        true_spells = kwargs["true_spells"] # spell.spelled, spell.correct
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

    def __str__(self):
        return 'AggregatedCandidator ' + str([str(candidator) for candidator in self._candidators])

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for candidator in self._candidators:
            candidates = candidator.get_candidates(text, spelled_words, **kwargs)
            for i, candidate in enumerate(candidates):
                all_candidates[i].extend(candidate)

        for i in range(len(all_candidates)):
            all_candidates[i] = list(set(all_candidates[i]))

        return all_candidates

    def get_candidates_by_candidator(self, text: str, spelled_words: List[SpelledWord], candidator_ind: int = 0, **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        candidator = self._candidators[candidator_ind]
        candidates = candidator.get_candidates(text, spelled_words, **kwargs)
        for i, candidate in enumerate(candidates):
            all_candidates[i].extend(candidate)

        return all_candidates

# Зачем мы берем совпадающий префикс?
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

    def __str__(self):
        return 'HunspellCandidator'

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for i, spelled_word in enumerate(spelled_words):
            candidates = self._hunspell.suggest(spelled_word.word)
            all_candidates[i] = list(candidates)
        return all_candidates


class SymSpellCandidator(BaseCandidator):
    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7, count_threshold=1):
        self.max_dictionary_edit_distance=max_dictionary_edit_distance
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_dictionary_edit_distance, prefix_length=prefix_length, count_threshold=count_threshold)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def __str__(self):
        return f'SymSpellCandidator MaxEditDist={self.max_dictionary_edit_distance}'

    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for i, spelled_word in enumerate(spelled_words):
            candidates = self.sym_spell.lookup(spelled_word.word, Verbosity.ALL, transfer_casing=True)
            all_candidates[i] = []
            for cand in candidates:
                all_candidates[i].append(cand.term)
        return all_candidates

# это основной класс для генерации кандидатов с помощью трансформера
class NNCandidator(BaseCandidator):
    def __init__(self, num_beams: int = 5):
        # это основной член класс который будет на генерить кандидатов
        self.gen = SwapWordGenerator("facebook/bart-base", torch.device("cpu"), num_beams=num_beams)
        # число гипотез при поиске кандидатов
        self.num_beams = num_beams

    def __str__(self):
        return f'NNCandidator num_beams={self.num_beams}'

    # основной метод класса который вернет кандидатов для каждого слова которое мы хотим переписать
    def get_candidates(self, text: str, spelled_words: List[SpelledWord], **kwargs) -> List[List[str]]:
        all_candidates: List[List[str]] = [[] for _ in spelled_words]
        for i, spelled_word in enumerate(spelled_words):
            # это какой-то бред… я зачем-то приписываю пробел в начало всего текста а не текущего слова - фигня
            # if spelled_word.interval[0] > 0 and spelled_word.text[spelled_word.interval[0] - 1] == ' ':
            #     all_candidates[i] = self.gen.generate(' ' + spelled_word.text, (spelled_word.interval[0] - 1, spelled_word.interval[1]))
            if spelled_word.interval[0] > 0 and spelled_word.text[spelled_word.interval[0] - 1] == ' ':
                all_candidates[i] = self.gen.generate(spelled_word.text, (spelled_word.interval[0] - 1, spelled_word.interval[1]))
            else:
                all_candidates[i] = self.gen.generate(spelled_word.text, (spelled_word.interval[0], spelled_word.interval[1]))

        return all_candidates


def main():
    # candidator = SymSpellCandidator(max_err=1, index_prefix_len=1)
    # candidator = SymSpellCandidator()
    candidator = NNCandidator()
    # candidator = AggregatedCandidator([HunspellCandidator(), SymSpellCandidator()])
    text = 'hello i am frim paris'
    sw = SpelledWord(text, (11, 15))
    print(sw.word)
    all_candidates = candidator.get_candidates(text=text, spelled_words=[sw])
    print(all_candidates)


if __name__ == '__main__':
    main()
