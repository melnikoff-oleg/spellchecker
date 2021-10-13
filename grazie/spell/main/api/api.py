import os
from typing import List, Optional

from grazie.common.main.ranking.catboost_ranker import CatBoostRanker
from grazie.spell.main.model.candidator import HunspellCandidator
from grazie.spell.main.model.detector import HunspellDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector
from grazie.spell.main.model.ranker import FeaturesSpellRanker
from grazie.spell.main.model.spellcheck_model import SpellCheckResult, SpellCheckModel


class ServerSpellApi:
    # in our case name is path to directory with model
    _name = "local_based"  # hunspell_cb_1
    _model: SpellCheckModel
    _round_digits: int = 4

    @staticmethod
    def create_model(path: str) -> SpellCheckModel:
        detector = HunspellDetector()
        candidator = HunspellCandidator()
        features_collector = FeaturesCollector.load(path)
        ranker = CatBoostRanker().load(os.path.join(path, "ranker.model"))
        model = SpellCheckModel(detector, candidator, FeaturesSpellRanker(features_collector, ranker))
        return model

    @staticmethod
    def download_model(name: Optional[str] = None):
        name = name or ServerSpellApi._name
        # load_aws_model(name)

    @staticmethod
    def initialize(name: Optional[str] = None, round_digits: int = 4):
        ServerSpellApi._round_digits = round_digits
        ServerSpellApi._name = name or ServerSpellApi._name
        # path = load_aws_model(ServerSpellApi._name)
        path = ServerSpellApi._name
        ServerSpellApi._model = ServerSpellApi.create_model(path)

    @staticmethod
    def check(texts: List[str], max_count: Optional[int] = None, round_digits: Optional[int] = None) -> List[List[SpellCheckResult]]:
        round_digits = round_digits or ServerSpellApi._round_digits
        return [ServerSpellApi._model.check(text, max_count, round_digits) for text in texts]
