import re
from abc import ABC, abstractmethod
from typing import List

import nltk
from nltk.corpus import words as nltk_words

from hunspell import Hunspell

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.preprocessing.tokenizer import SyntokTextTokenizer


class BaseDetector(ABC):
    def __init__(self):
        self._is_word = re.compile("[a-zA-Z]+'?[a-zA-Z]*")

    def is_word(self, word: str) -> bool:
        return bool(self._is_word.match(word))

    @abstractmethod
    def detect(self, text: str, **kwargs) -> List[SpelledWord]:
        raise NotImplementedError

# Разве норм что тут везде ищется именно первое вхождение слова в текст? если есть 2 одинаковых ошибки
# Вторая останется пропущенной
class IdealDetector(BaseDetector):
    def detect(self, text: str, **kwargs) -> List[SpelledWord]:
        true_spells = kwargs["true_spells"]

        res = []
        fict_text = text

        for spell in true_spells:
            start = fict_text.index(spell.spelled)
            finish = start + len(spell.spelled)
            # mark this occurrence of word
            gap = ''.join('#' for ind in range(start, finish))
            fict_text = fict_text[:start] + gap + fict_text[finish:]
            res.append(SpelledWord(text, (start, finish)))

        return res


class WordBaseDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self._tokenizer = SyntokTextTokenizer()

    def detect(self, text: str, **kwargs) -> List[SpelledWord]:
        fict_text = text
        intervals = []
        words = self._tokenizer.tokenize(text)

        # Тут тоже только первое вхождение ошибки
        for i, word in enumerate(words):
            if self.is_spelled(word):
                start = fict_text.find(word)
                finish = start + len(word)
                assert fict_text[start:finish] == word
                # mark this occurrence of word
                gap = ''.join('#' for ind in range(start, finish))
                fict_text = fict_text[:start] + gap + fict_text[finish:]
                intervals.append(SpelledWord(text, (start, finish)))

        return intervals

    def is_spelled(self, word: str) -> bool:
        raise NotImplementedError

# Тупо проверяем есть ли слово в словаре
class DictionaryDetector(WordBaseDetector):
    def __init__(self):
        super().__init__()
        self.require_nltk()
        self._correct_words = set(word.lower() for word in nltk_words.words())

    @staticmethod
    def require_nltk():
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')

    def is_spelled(self, word: str) -> bool:
        low_word = word.lower()
        spelled = self.is_word(low_word) and low_word not in self._correct_words
        return spelled


class HunspellDetector(WordBaseDetector):
    def __init__(self):
        super().__init__()
        self._hunspell = Hunspell()

    def is_spelled(self, word: str) -> bool:
        spelled = self.is_word(word) and not self._hunspell.spell(word)
        return spelled
