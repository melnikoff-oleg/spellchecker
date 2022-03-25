import datetime
import itertools
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
    LevenshteinCandidator, HunspellCandidator, SymSpellCandidator, NNCandidator
from grazie.spell.main.model.detector import IdealDetector, DictionaryDetector, HunspellDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector
from grazie.spell.main.model.ranker import FeaturesSpellRanker
from grazie.spell.main.model.spellcheck_model import SpellCheckModel, SpellCheckModelCharBasedTransformerMedium, SpellCheckModelNeuSpell, SpellCheckModelCharBasedTransformerSmall


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


def sort_experiments(experiment_save_path):
    if exists(experiment_save_path):
        with open(experiment_save_path) as f:
            exp_res_dict = json.load(f)
        arr_to_sort = []
        for ind, exp in enumerate(exp_res_dict):
            arr_to_sort.append([exp['Pipeline Results']['All Together']['acc@1'], ind])
        arr_to_sort.sort(reverse=True)
        new_exp_dict = []
        for val in arr_to_sort:
            new_exp_dict.append(exp_res_dict[val[1]])
        with open(experiment_save_path, 'w') as f:
            json.dump(new_exp_dict, f, indent=4)


def get_time_diff(start):
    return str(datetime.timedelta(seconds=round(time.time() - start)))


def save_experiment_results(detector_name, candidator_name, ranker_name, ranker_features, dataset_name, train_data_len, test_data_len, pipeline_metrics, pipeline_mistakes=(), data_prep_time: str = '0', ranker_train_time: str = '0', pipeline_eval_time: str = '0', experiment_save_path: str = 'junk_experiments.json', save_experiment: bool = False, rewrite: bool = True):
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    experiment_packed = {'Date': dt_string,
                         'Model Config': {
                             'Detector': detector_name, 'Candidator': candidator_name,
                             'Ranker': ranker_name, 'Features': {'RankerFeatures': ranker_features}
                         },
                         'Dataset': {'Dataset Name': dataset_name, 'Dataset Size': train_data_len + test_data_len,
                                     'Train Size': train_data_len, 'Test Size': test_data_len},
                         'Pipeline Results': {
                             'All Together': {
                                 'Word-level accuracy': pipeline_metrics['Word-level accuracy'],
                                 'Precision': pipeline_metrics['Precision'],
                                 'Recall': pipeline_metrics['Recall'],
                                 'F_0.5': pipeline_metrics['F_0.5'],
                                 'acc@1': pipeline_metrics['pipeline_acc@1'],
                                 'acc@3': pipeline_metrics['pipeline_acc@3'],
                             },
                             'Detector': {
                                 'Precision': pipeline_metrics['detector_precision'],
                                 'Recall': pipeline_metrics['detector_recall'],
                                 'Mistakes Not Found': pipeline_mistakes['mistake_not_found'],
                                 'False Detection': pipeline_mistakes['false_detection']
                             },
                             'Candidator': {
                                 'acc@inf': pipeline_metrics['candidator_acc (acc@inf)'],
                                 'Correct Cand Not Found': pipeline_mistakes['correct_cand_not_found']
                             },
                             'Ranker': {
                                 'acc@1': pipeline_metrics['ranker_acc@1'],
                                 'acc@3': pipeline_metrics['ranker_acc@3'],
                                 'Ranking Mistake': pipeline_mistakes['ranking_mistake']
                             }
                         },
                         'Runtime': {
                             'Data Preparation': data_prep_time,
                             'Ranker Train': ranker_train_time,
                             'Pipeline Eval': pipeline_eval_time,
                         }
                         }

    print(experiment_packed)

    if save_experiment:
        if exists(experiment_save_path) and not rewrite:
            with open(experiment_save_path) as f:
                exp_res_dict = json.load(f)
        else:
            exp_res_dict = []
        exp_res_dict.append(experiment_packed)
        with open(experiment_save_path, 'w') as f:
            json.dump(exp_res_dict, f)
        sort_experiments(experiment_save_path)


def train_model(detector, candidator, ranker, ranker_features, train_data: List[SpelledText],
                test_data: List[SpelledText], freqs_path: str, bigrams_path: str, trigrams_path: str,
                path_save_exp: str, dataset_name: str, save_experiment: bool = True) -> None:
    start = time.time()
    features_collector = FeaturesCollector(ranker_features, bigrams_path, trigrams_path,
                                           FeaturesCollector.load_freqs(freqs_path))
    train_rank_data = prepare_ranking_training_data(train_data, candidator, features_collector)
    test_rank_data = prepare_ranking_training_data(test_data, candidator, features_collector)
    data_prep_time = get_time_diff(start)

    start = time.time()
    ranker.fit(train_rank_data, test_rank_data, epochs=20, lr=3e-4, l2=0., l1=0.)
    ranker_train_time = get_time_diff(start)
    print("Ranker's feature importancies:\n", ranker.get_feature_importance(train_rank_data, ranker_features))

    start = time.time()
    model = SpellCheckModel(detector, candidator, FeaturesSpellRanker(features_collector, ranker))
    print("Evaluate all")
    pipeline_metrics, pipeline_mistakes = evaluate(model, test_data, verbose=True, path_save_result=path_save_exp + 'result.txt', max_mistakes_log=100)
    print('LOL', pipeline_metrics, '\n\n\n', pipeline_mistakes, 'LOL')

    print()
    pipeline_eval_time = get_time_diff(start)

    detector_name = type(detector).__name__
    candidator_name = str(candidator)
    ranker_name = type(ranker).__name__
    train_data_len = len(train_data)
    test_data_len = len(test_data)

    save_experiment_results(detector_name, candidator_name, ranker_name, ranker_features, dataset_name, train_data_len, test_data_len, pipeline_metrics, pipeline_mistakes, data_prep_time, ranker_train_time, pipeline_eval_time, path_save_exp + 'report.json', save_experiment)


def eval_e2e_model(model, test_data, dataset_name, path_save_exp):
    pipeline_metrics, pipeline_mistakes = evaluate(model, test_data, verbose=True, max_mistakes_log=20, path_save_result=path_save_exp + 'result_test.txt')
    save_experiment_results(str(model), str(model), str(model), [], dataset_name, 1300000,
                            len(test_data), pipeline_metrics, pipeline_mistakes, experiment_save_path=path_save_exp + 'report_test.json', save_experiment=True)


def main():
    path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
    # path_prefix = '/home/ubuntu/omelnikov/'
    train_gt = path_prefix + 'grazie/spell/main/data/datasets/1blm/1blm.train.gt'
    train_noise = path_prefix + 'grazie/spell/main/data/datasets/1blm/1blm.train.noise'
    test_gt = path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.gt'
    test_noise = path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.noise'
    path_save_exp = path_prefix + '/grazie/spell/main/data/experiments/neuspell_bert/'
    freqs_table_path = path_prefix + 'grazie/spell/main/data/n_gram_freqs/1_grams.csv'
    bigrams_table_path = path_prefix + 'grazie/spell/main/data/n_gram_freqs/2_grams.csv'
    trigrams_table_path = path_prefix + 'grazie/spell/main/data/n_gram_freqs/3_grams.csv'
    dataset_name = test_gt.split('/')[-1]
    train_data, test_data_fict = get_test_data(train_gt, train_noise, size=500, train_part=1.0)
    test_data, test_data_fict = get_test_data(test_gt, test_noise, size=100000, train_part=1.0)


    # eval old BART
    # train_model(HunspellDetector(), HunspellCandidator(), CatBoostRanker(iterations=100), ['bart_prob', 'levenshtein'], train_data, test_data, freqs_table_path, bigrams_table_path,
    #             trigrams_table_path, path_save_exp, dataset_name, save_experiment=True)

    # eval char-based-transformer SMALL
    # char_based_transformer = SpellCheckModelCharBasedTransformerMedium(
    #     checkpoint=path_prefix + 'grazie/spell/main/training/model_small_2_4.pt')
    # eval_e2e_model(char_based_transformer, test_data, dataset_name, path_save_exp)


    # eval char-based-transformer MEDIUM
    # char_based_transformer = SpellCheckModelCharBasedTransformerMedium(
    #     checkpoint=path_prefix + 'grazie/spell/main/training/model_sch_lin_warm_239_2.pt')
    #
    # eval_e2e_model(char_based_transformer, test_data, dataset_name, path_save_exp)

    # eval T5
    # model = SpellCheckModelT5()
    # eval_e2e_model(model, test_data, dataset_name, path_save_exp)

    # eval NeuSpell
    model = SpellCheckModelNeuSpell()
    eval_e2e_model(model, test_data, dataset_name, path_save_exp)


if __name__ == '__main__':
    main()
