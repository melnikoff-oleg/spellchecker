from typing import List

from tqdm import tqdm

from grazie.common.main.ranking.catboost_ranker import CatBoostRanker
from grazie.common.main.ranking.ranker import RankQuery, RankVariant, Ranker
from grazie.spell.main.data.base import SpelledText
from grazie.spell.main.data.utils import default_args_parser, get_test_data
from grazie.spell.main.evaluation.evaluate import evaluate, evaluate_ranker
from grazie.spell.main.model.candidator import BaseCandidator, HunspellCandidator, AggregatedCandidator, IdealCandidator
from grazie.spell.main.model.detector import HunspellDetector, IdealDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector
from grazie.spell.main.model.ranker import FeaturesSpellRanker
from grazie.spell.main.model.spellcheck_model import SpellCheckModel


def prepare_ranking_training_data(spell_data: List[SpelledText], candidator: BaseCandidator,
                                  features_collector: FeaturesCollector) -> List[RankQuery]:
    labeled_data = []
    idx = 0
    ideal_detector = IdealDetector()
    candidator = AggregatedCandidator([candidator, IdealCandidator()])

    for spelled_text in tqdm(spell_data):
        text = spelled_text.text
        spells = spelled_text.spells
        spelled_words = ideal_detector.detect(text, true_spells=spells)
        all_candidates = candidator.get_candidates(text, spelled_words, true_spells=spells)

        for i, (spelled_word, candidates, spell) in enumerate(zip(spelled_words, all_candidates, spells)):
            all_features = features_collector.collect(text, spelled_word, candidates)
            variants = []
            for candidate, features in zip(candidates, all_features):
                target = int(candidate == spell.correct)
                variants.append(RankVariant(features, target))
            labeled_data.append(RankQuery(idx, variants))
            idx += 1

    return labeled_data


def train_ranker(train_data: List[SpelledText], test_data: List[SpelledText], freqs_path: str) -> Ranker:
    # detector = DictionaryDetector()
    # candidator = LevenshteinCandidator(max_err=2, index_prefix_len=2)
    detector = HunspellDetector()
    candidator = HunspellCandidator()
    features_collector = FeaturesCollector(["levenshtein", "freq", "soundex", "metaphone"],
                                           FeaturesCollector.load_freqs(freqs_path))

    train_rank_data = prepare_ranking_training_data(train_data, candidator, features_collector)
    test_rank_data = prepare_ranking_training_data(test_data, candidator, features_collector)

    # ranker = NeuralRanker()
    ranker = CatBoostRanker(iterations=100)
    ranker.fit(train_rank_data, test_rank_data, epochs=20, lr=3e-4, l2=0., l1=0.)
    # ranker.save(save_path)

    evaluate_ranker(FeaturesSpellRanker(features_collector, ranker), test_data, candidator=candidator, verbose=True)
    print("Evaluate ranker")
    print()

    model = SpellCheckModel(detector, candidator, FeaturesSpellRanker(features_collector, ranker))
    print("Evaluate all")
    evaluate(model, test_data, verbose=True)

    return ranker


def parse_args():
    parser = default_args_parser()

    parser.add_argument("--freqs_path", type=str, required=True, help="Path to freqs dictionary")
    parser.add_argument("--ranker_path", type=str, required=True, help="Saved CatBoost ranker")
    # parser.add_argument("--seed", type=int, required=False, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    # fix_seed(args.seed)

    train_data, test_data = get_test_data(args.texts_path, args.size)

    ranker = train_ranker(train_data, test_data, args.freqs_path)
    ranker.save(args.ranker_path)


if __name__ == '__main__':
    main()
