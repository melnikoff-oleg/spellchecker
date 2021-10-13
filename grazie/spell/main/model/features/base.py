from abc import ABC, abstractmethod
from typing import List, Tuple

from transformers import AutoTokenizer

from grazie.spell.main.model.base import SpelledWord


class BaseFeature(ABC):
    def compute(self, text: str, spelled_word: SpelledWord, candidate: str) -> float:
        return self.compute_candidates(text, spelled_word, [candidate])[0]

    @abstractmethod
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        raise NotImplementedError


class BaseTransformerFeature(BaseFeature, ABC):
    def __init__(self, name: str = "bert-base-uncased"):
        self._name = name
        self._tokenizer = AutoTokenizer.from_pretrained(self._name)

    def prepare(self, text, spelled_word: SpelledWord, candidates: List[str]) -> Tuple[List[List[int]], List[int], int]:
        masked_text = text.replace(spelled_word.word, "[MASK]")
        # maybe we should replace " " + spelled_word to solve problem with two consecutive spaces?
        masked_token_ids = self._tokenizer(masked_text)['input_ids']

        texts = [text.replace(spelled_word.word, candidate) for candidate in candidates]
        candidates_ids = self._tokenizer(texts, padding=True)
        first_candidate = candidates_ids["input_ids"][0]

        left_idx = 0
        while first_candidate[left_idx] == masked_token_ids[left_idx]:
            left_idx += 1

        return candidates_ids, masked_token_ids, left_idx
