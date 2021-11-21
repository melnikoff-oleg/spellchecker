import datetime
import json
import time
from os.path import exists
from typing import List

from tqdm import tqdm

from grazie.common.main.ranking.catboost_ranker import CatBoostRanker
from grazie.common.main.ranking.ranker import RankQuery, RankVariant
from grazie.spell.main.data.base import SpelledText
from grazie.spell.main.data.utils import get_test_data
from grazie.spell.main.evaluation.evaluate import evaluate, evaluate_ranker
from grazie.spell.main.model.candidator import BaseCandidator, AggregatedCandidator, IdealCandidator, \
    LevenshteinCandidator, HunspellCandidator
from grazie.spell.main.model.detector import IdealDetector, DictionaryDetector, HunspellDetector
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


def sort_experiments():
    experiment_save_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/experiments/experiments.json'
    if exists(experiment_save_path):
        with open(experiment_save_path) as f:
            exp_res_dict = json.load(f)
        arr_to_sort = []
        for ind, exp in enumerate(exp_res_dict):
            arr_to_sort.append([exp['experiment_results']["Pipeline Metrics"]["acc@1"], ind])
        arr_to_sort.sort(reverse=True)
        new_exp_dict = []
        for val in arr_to_sort:
            new_exp_dict.append(exp_res_dict[val[1]])
        with open(experiment_save_path, 'w') as f:
            json.dump(new_exp_dict, f, indent=4)


def get_time_diff(start):
    return str(datetime.timedelta(seconds=round(time.time() - start)))


def train_model(detector, candidator, ranker, ranker_features, train_data: List[SpelledText], test_data: List[SpelledText], freqs_path: str, experiment_save_path: str, dataset_name: str, save_experiment: bool = True) -> None:

    start = time.time()
    features_collector = FeaturesCollector(ranker_features, FeaturesCollector.load_freqs(freqs_path))
    train_rank_data = prepare_ranking_training_data(train_data, candidator, features_collector)
    test_rank_data = prepare_ranking_training_data(test_data, candidator, features_collector)
    data_prep_time = get_time_diff(start)

    start = time.time()
    ranker.fit(train_rank_data, test_rank_data, epochs=20, lr=3e-4, l2=0., l1=0.)
    ranker_train_time = get_time_diff(start)
    print("Ranker's feature importancies:\n", ranker.get_feature_importance(train_rank_data, ranker_features))

    start = time.time()
    print("Evaluate ranker")
    ranker_metrics, ranker_mistakes = evaluate_ranker(FeaturesSpellRanker(features_collector, ranker), test_data, candidator=candidator, verbose=True)
    print()
    ranker_eval_time = get_time_diff(start)

    start = time.time()
    model = SpellCheckModel(detector, candidator, FeaturesSpellRanker(features_collector, ranker))
    print("Evaluate all")
    pipeline_metrics, pipeline_mistakes = evaluate(model, test_data, verbose=True)
    print()
    pipeline_eval_time = get_time_diff(start)

    experiment_results = {'Detector': type(detector).__name__, 'Candidator': type(candidator).__name__, 'Ranker':  type(ranker).__name__, 'Features': {'RankerFeatures': ranker_features}, 'Dataset': dataset_name, 'Dataset Size': len(train_data) + len(test_data),  'Ranker Metrics': ranker_metrics, 'Ranker Mistakes': ranker_mistakes, 'Pipeline Metrics': pipeline_metrics, 'Pipeline Mistakes': pipeline_mistakes, 'Data Preparation Time': data_prep_time, 'Ranker Train Time': ranker_train_time, 'Ranker Eval Time': ranker_eval_time, 'Pipeline Eval Time': pipeline_eval_time}
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment_packed = {'name': type(detector).__name__[:4] + type(candidator).__name__[:4] + type(ranker).__name__[:4], 'date': dt_string, 'experiment_results': experiment_results}
    if exists(experiment_save_path):
        with open(experiment_save_path) as f:
            exp_res_dict = json.load(f)
    else:
        exp_res_dict = []
    exp_res_dict.append(experiment_packed)
    if save_experiment:
        with open(experiment_save_path, 'w') as f:
            json.dump(exp_res_dict, f)
    sort_experiments()
    print(experiment_results)


def main():

    gt_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea4k'
    noise_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea4k.noise'
    freqs_table_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/1_grams.csv'
    # model_save_path = '/Users/olegmelnikov/Downloads/ranker_model'
    experiment_save_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/experiments/experiments.json'
    dataset_name = gt_texts_path.split('/')[-1]
    train_data, test_data = get_test_data(gt_texts_path, noise_texts_path, size=500)

    detectors = [HunspellDetector(), DictionaryDetector()]
    candidators = [HunspellCandidator(), LevenshteinCandidator(max_err=2, index_prefix_len=2)]
    # rankers = [CatBoostRanker(iterations=10), CatBoostRanker(iterations=100), CatBoostRanker(iterations=1000)]
    features = ["bart_prob", "bert_prob", "suffix_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "jaro_winkler", "freq", "log_freq", "sqrt_freq", "soundex", "metaphone", "keyboard_dist", "cands_less_dist"]

    detector = HunspellDetector()
    candidator = HunspellCandidator()
    ranker = CatBoostRanker(iterations=100)
    ranker_features = [
        ['bart_prob'],
        # ["bart_prob", "bert_prob"],
        # ["bart_prob", "bert_prob", "bigram_freq", "soundex", "freq"],
        # ["bart_prob", "bert_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq", "soundex", "metaphone", "keyboard_dist", "cands_less_dist"],
        # ["bart_prob", "bert_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq",
        #  "soundex", "metaphone", "keyboard_dist"],
        # ["bart_prob", "cand_length_diff", "bigram_freq", "levenshtein", "freq"],
        # ["bart_prob", "cand_length_diff", "init_word_length", "levenshtein", "freq"],
        # ["bart_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq",
        #  "soundex", "metaphone", "keyboard_dist"],
        # ["bart_prob", "bigram_freq", "cand_length_diff", "levenshtein", "freq", "keyboard_dist"],
        # ["bart_prob", "cand_length_diff", "levenshtein", "freq", "metaphone", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length"],
        # ["suffix_prob", "cand_length_diff", "levenshtein", "freq"],
        # ["bert_prob", "cand_length_diff", "levenshtein", "freq"],
        # ["bert_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq", "keyboard_dist"],
        # ["bert_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq", "soundex",
        #  "metaphone", "keyboard_dist"],
        # ["bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq", "keyboard_dist"],
        # ["bigram_freq", "cand_length_diff", "levenshtein", "freq"],
        # ["bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq", "soundex", "metaphone", "keyboard_dist"],
        # ["levenshtein", "log_freq", "soundex", "cands_less_dist"],
        # ["levenshtein", "sqrt_freq", "soundex", "metaphone", "keyboard_dist"],
        # ["levenshtein", "freq", "cands_less_dist", "metaphone"],
        # ["levenshtein", "freq", "soundex", "metaphone"],
        # ["bart_prob", "bert_prob", "bigram_freq", "trigram_freq", "cand_length_diff", "init_word_length", "levenshtein", "freq",
        #  "soundex", "metaphone", "keyboard_dist"],
    ]
    # for ranker in rankers:
    for rf in ranker_features:
        train_model(detector, candidator, ranker, rf, train_data, test_data, freqs_table_path, experiment_save_path, dataset_name, save_experiment=True)


if __name__ == '__main__':
    main()
