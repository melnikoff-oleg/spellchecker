import re
from abc import ABC, abstractmethod
from typing import List

import nltk
from nltk.corpus import words as nltk_words

from hunspell import Hunspell

from model.base import SpelledWord
from data_utils.tokenizer import SyntokTextTokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch


class BaseDetector(ABC):
    def __init__(self):
        self._is_word = re.compile("[a-zA-Z]+'?[a-zA-Z]*")

    def is_word(self, word: str) -> bool:
        return bool(self._is_word.match(word))

    @abstractmethod
    def detect(self, text: str, **kwargs) -> List[SpelledWord]:
        raise NotImplementedError


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


class BERTDetector(BaseDetector):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        # model_checkpoint = '/home/ubuntu/omelnikov/distilbert-base-uncased-finetuned-tagging/checkpoint-124500'\
        model_checkpoint = '/home/ubuntu/omelnikov/spellchecker/training/checkpoints/BERT-detector-2-epochs'
        self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        tokenizer_checkpoint = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.threshold = threshold

    def detect(self, text: str, **kwargs) -> List[SpelledWord]:

        word_tokens = nltk.word_tokenize(text)
        tokenized_inputs = self.tokenizer(word_tokens, truncation=True, is_split_into_words=True,
                                          return_tensors='pt').to(self.device)["input_ids"]
        result = self.model(tokenized_inputs)
        # labels = torch.argmax(result.logits, dim=2).to('cpu').tolist()[0]
        labels_threshold = torch.softmax(result.logits.to('cpu')[0], dim=1)[:, 1] > self.threshold

        tokenized_input = self.tokenizer(word_tokens, is_split_into_words=True)
        word_ids = tokenized_input.word_ids()

        # DEBUG
        # print('Word ids:', word_ids)
        # print('Logits:', result.logits.to('cpu'))
        # print('Probs:', torch.softmax(result.logits.to('cpu')[0], dim=1))
        # print('Diffs:', torch.softmax(result.logits.to('cpu')[0], dim=1)[:, 1] > threshold)
        # print('Labels:', labels)
        # print()

        intervals = []

        def spans(txt):
            tokens = nltk.word_tokenize(txt)
            offset = 0
            for token in tokens:
                offset = txt.find(token, offset)
                yield offset, offset + len(token)
                offset += len(token)

        word_spans = list(spans(text))

        final_labels = [0 for i in word_tokens]
        for ind in range(len(labels_threshold)):
            if labels_threshold[ind]:
                final_labels[word_ids[ind]] = 1

        for ind, i in enumerate(final_labels):
            if i == 1:
                intervals.append(SpelledWord(text, word_spans[ind]))

        # DEBUG
        # print(f'Final labels:', final_labels)
        # print(f'Intervals:', intervals)

        return intervals


class WordBaseDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self._tokenizer = SyntokTextTokenizer()

    def detect(self, text: str, **kwargs) -> List[SpelledWord]:
        fict_text = text
        intervals = []
        words = self._tokenizer.tokenize(text)

        # single quote handle
        real_words = []
        for i, word in enumerate(words):
            if i < len(words) - 1 and words[i + 1] in ["'re", "'ve", "'s", "'t", "n't", "'d"]:
                real_words.append(word + words[i + 1])
                continue
            if word in ["'re", "'ve", "'s", "'t", "n't", "'d"]:
                continue
            real_words.append(word)
        words = real_words

        # Тут тоже только первое вхождение ошибки - Fixed
        cur_shift = 0
        for i, word in enumerate(words):
            start = fict_text.find(word, cur_shift)
            if self.is_spelled(word):
                finish = start + len(word)
                assert fict_text[start:finish] == word
                # mark this occurrence of word
                gap = ''.join('#' for ind in range(start, finish))
                fict_text = fict_text[:start] + gap + fict_text[finish:]
                intervals.append(SpelledWord(text, (start, finish)))
            cur_shift += len(word)
            cur_shift = max(cur_shift, start + len(word))

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
        try:
            spelled = self.is_word(word) and not self._hunspell.spell(word)
        except Exception as e: # for encoding exceptions
            print('Word with exception:', word)
            spelled = False
        return spelled


def test():
    bert_detector = BERTDetector()
    sent = 'Hello we arre th company of frineds'
    print('Test sentence:', sent)
    bert_detector.detect(sent)


if __name__ == '__main__':
    test()
    # h = HunspellDetector()
    # print(h.detect("On the we're, being rich and famous doesn't always bring happeness, whereas the majority of the population wish they were rich and famous."))
    # tokenizer = SyntokTextTokenizer()
    # print(tokenizer.tokenize("On the we're, being rich and famous doesn't always bring happeness, whereas the majority of the population wish they were rich and famous."))

    # h = Hunspell()
    # print(h.spell("don't"))
