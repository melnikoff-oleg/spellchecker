from abc import ABC, abstractmethod
from typing import List, Optional

import attr


@attr.s(auto_attribs=True, frozen=True)
class RankVariant:
    features: List[float]
    target: float


@attr.s(auto_attribs=True, frozen=True)
class RankQuery:
    id: int
    variants: List[RankVariant]


@attr.s(auto_attribs=True, frozen=True)
class RankResult:
    id: int
    variants: List[RankVariant]
    scores: List[float]


class Ranker(ABC):
    @abstractmethod
    def predict(self, features: List[List[float]]) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_data: List[RankQuery], test_data: List[RankQuery], **kwargs) -> 'Ranker':
        raise NotImplementedError

    @abstractmethod
    def importance_info(self, train_data: List[RankQuery]):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> 'Ranker':
        raise NotImplementedError


def ranked_queries(qs: List[RankQuery], model: Optional[Ranker] = None) -> List[RankResult]:
    rank_results = []

    for query in qs:
        if model is not None:
            features = [sugg.features for sugg in query.variants]
            preds = model.predict(features)
        else:
            preds = [1.0 - i for i in range(len(query.variants))]

        ranked_query = RankResult(query.id, query.variants, preds)
        rank_results.append(ranked_query)

    return rank_results
