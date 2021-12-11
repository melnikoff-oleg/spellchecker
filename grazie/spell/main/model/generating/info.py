import numpy as np
from typing import List

# список вероятностей каждого токена для генерации
class GenerationInfo:
    def __init__(self, probs: List[float] = None, score: float = None, ids: List[int] = None):
        self._probs = probs if probs is not None else []
        self.ids = ids or []
        self._score = score

    def add(self, p: float):
        self._probs.append(p)

    def trim(self, left, right=None) -> 'GenerationInfo':
        if right is None:
            right = left
            left = 0
        self._probs = self._probs[left:right]
        self.ids = self.ids[left:right]
        return self

    def probs(self) -> List[float]:
        return self._probs

    def score(self) -> float:
        if self._score is None:
            return float(np.mean([np.log(p) for p in self._probs]))
        return self._score

    def to_dict(self):
        return {"probs": self._probs, "ids": self.ids, "score": self._score}

    @staticmethod
    def from_dict(dict):
        return GenerationInfo(dict["probs"], dict.get("score", None), dict.get("ids", None))
