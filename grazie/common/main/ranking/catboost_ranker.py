import copy
from typing import List

from catboost import CatBoost, Pool

from grazie.common.main.ranking.ranker import Ranker, RankQuery


class CatBoostRanker(Ranker):
    def __init__(self, **kwargs):
        self.default_parameters = {
            'iterations': 10,
            'loss_function': 'PairLogit',
            'custom_metric': ['PrecisionAt:top=1', 'PrecisionAt:top=3', 'NDCG', 'PFound', 'AverageGain:top=10'],
            'verbose': False,
            'random_seed': 0,
        }

        parameters = copy.deepcopy(self.default_parameters)
        for k, v in kwargs.items():
            parameters[k] = v

        self.model = CatBoost(parameters)

    def predict(self, features: List[List[float]]) -> List[float]:
        if not features:
            return []
        pool = Pool(features)
        return self.model.predict(pool, prediction_type='Probability')[:, 1].tolist()

    def get_feature_importance(self, train_data: List[RankQuery], feartures_names: List[str]):
        train_pool = create_pool(train_data)
        feature_importance = self.model.get_feature_importance(data=train_pool)
        fi_dict = {}
        for fn, fi in zip(feartures_names, feature_importance):
            fi_dict[fn] = fi
        return {k: v for k, v in sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)}

    def fit(self, train_data: List[RankQuery], test_data: List[RankQuery], **kwargs) -> 'CatBoostRanker':
        train_pool = create_pool(train_data)
        test_pool = create_pool(test_data)
        self.model.fit(train_pool, eval_set=test_pool, plot=False, verbose=False)
        return self

    def importance_info(self, train_data: List[RankQuery]):
        train_pool = create_pool(train_data)
        print("Features importance")
        for name, imp in zip(self.model.feature_names_, self.model.get_feature_importance(train_pool)):
            print(name, imp)
        print()

    def save(self, path: str):
        self.model.save_model(path, format="onnx")

    def load(self, path: str) -> 'CatBoostRanker':
        self.model.load_model(path, format="onnx")
        return self


def create_pool(data: List[RankQuery]):
    features = []
    labels = []
    groups = []
    for query in data:
        for v in query.variants:
            features.append(v.features)
            labels.append(v.target)
            groups.append(query.id)
    return Pool(data=features, label=labels, group_id=groups)
