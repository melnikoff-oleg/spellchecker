from sklearn.linear_model import LogisticRegression
import pickle
import attr
from model.candidator import *
from model.ranking_utils.features_collector import FeaturesCollector
from tqdm import tqdm
from data_utils.utils import get_texts_from_file
import numpy as np
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


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
    def fit(self, train_data: List[RankQuery], test_data: List[RankQuery]):
        raise NotImplementedError

    @abstractmethod
    def importance_info(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError


class LogisticRegressionRanker(Ranker):
    def __init__(self):
        self.model = LogisticRegression(random_state=0)

    def predict(self, features: List[List[List[float]]]) -> List[List[float]]:
        res = []
        for cur_features in features:
            res.append(self.model.predict_proba(cur_features)[:, 1])
        return res

    def fit(self, train_data: List[RankQuery], test_data: List[RankQuery]):
        X_train, y_train = [], []
        for rq in train_data:
            for variant in rq.variants:
                X_train.append(variant.features)
                y_train.append(variant.target)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test, y_test = [], []
        for rq in test_data:
            for variant in rq.variants:
                X_test.append(variant.features)
                y_test.append(variant.target)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print('X_train:', X_train)
        print('Shape:', X_train.shape)
        print('y_train:', y_train)
        print('Shape:', y_train.shape)

        self.model.fit(X_train, y_train)
        print('Accuracy of classification:', self.model.score(X_test, y_test))

    def importance_info(self):
        print(f'LogisticRegression coefficients: {self.model.coef_}, Intercept: {self.model.intercept_}')

    def save(self, path: str):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path: str):
        self.model = pickle.load(open(path, 'rb'))


@attr.s(auto_attribs=True, frozen=True)
class Spell:
    spelled: str
    correct: str


@attr.s(auto_attribs=True, frozen=True)
class SpelledText:
    text: str
    spells: List[Spell]


def prepare_ranking_training_data(features_collector: FeaturesCollector) -> List[RankQuery]:
    texts_noise, texts_gt = get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea500.noise'), \
                            get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea500.gt')
    labeled_data = []
    spelled_words = []
    gt_words = []
    for text_noise, text_gt in tqdm(zip(texts_noise, texts_gt), total=len(texts_noise)):
        words_noise, words_gt = text_noise.split(' '), text_gt.split(' ')
        if len(words_noise) != len(words_gt):
            continue
        cur_shift = 0
        for idx, (word_noise, word_gt) in enumerate(zip(words_noise, words_gt)):
            if word_noise != word_gt:
                start = cur_shift
                end = start + len(word_noise)
                spelled_word = SpelledWord(text_noise, interval=(start, end))
                spelled_words.append(spelled_word)
                gt_words.append(word_gt)
            cur_shift += len(word_noise) + 1
    candidates = HunspellCandidator().get_candidates('fictive string', spelled_words)
    all_features = features_collector.collect(spelled_words, candidates)

    # DEBUG
    # for i, j, k in zip(spelled_words, candidates, all_features):
    #     print(i)
    #     print(j)
    #     print(k)
    #     print()

    for idx in range(len(spelled_words)):
        variants = []
        for jdx in range(len(candidates[idx])):
            target = int(candidates[idx][jdx] == gt_words[idx])
            variants.append(RankVariant(all_features[idx][jdx], target))
        labeled_data.append(RankQuery(idx, variants))

    return labeled_data


def train():
    model = LogisticRegressionRanker()
    data_ranker_train = prepare_ranking_training_data(FeaturesCollector(features_names=['bart_prob', 'levenshtein']))
    model.fit(data_ranker_train, data_ranker_train)
    model.save(PATH_PREFIX + 'model/ranking_utils/oldbartLN_lev_ranker.pickle')
    model.load(PATH_PREFIX + 'model/ranking_utils/oldbartLN_lev_ranker.pickle')
    model.importance_info()


def test():
    data_ranker_train = prepare_ranking_training_data(FeaturesCollector(features_names=['bart_prob', 'levenshtein']))
    print(data_ranker_train)


def main():
    train()


if __name__ == '__main__':
    main()
