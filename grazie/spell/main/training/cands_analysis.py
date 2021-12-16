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

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.candidator import BaseCandidator, AggregatedCandidator, IdealCandidator, \
    LevenshteinCandidator, HunspellCandidator, SymSpellCandidator, NNCandidator
from grazie.spell.main.model.detector import IdealDetector, DictionaryDetector, HunspellDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector



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
            # arr_to_sort.append([exp['experiment_results']["Pipeline Metrics"]["acc@1"], ind])
            arr_to_sort.append([exp['Pipeline Results']['All Together']['acc@1'], ind])
        arr_to_sort.sort(reverse=True)
        new_exp_dict = []
        for val in arr_to_sort:
            new_exp_dict.append(exp_res_dict[val[1]])
        with open(experiment_save_path, 'w') as f:
            json.dump(new_exp_dict, f, indent=4)


def get_time_diff(start):
    return str(datetime.timedelta(seconds=round(time.time() - start)))


def save_experiment_results(dt_string: str, detector_name, candidator_name, ranker_name, ranker_features, dataset_name, train_data_len, test_data_len, pipeline_metrics, pipeline_mistakes, data_prep_time, ranker_train_time, pipeline_eval_time, experiment_save_path, save_experiment):
    experiment_packed = {'Date': dt_string,
                         'Model Config': {
                             'Detector': detector_name, 'Candidator': candidator_name,
                             'Ranker': ranker_name, 'Features': {'RankerFeatures': ranker_features}
                         },
                         'Dataset': {'Dataset Name': dataset_name, 'Dataset Size': train_data_len + test_data_len,
                                     'Train Size': train_data_len, 'Test Size': test_data_len},
                         'Pipeline Results': {
                             'All Together': {
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

    if exists(experiment_save_path):
        with open(experiment_save_path) as f:
            exp_res_dict = json.load(f)
    else:
        exp_res_dict = []
    exp_res_dict.append(experiment_packed)
    if save_experiment:
        with open(experiment_save_path, 'w') as f:
            json.dump(exp_res_dict, f)
    sort_experiments(experiment_save_path)
    print(experiment_packed)



def main():
    gt_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea500.clean'
    noise_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea500.clean.noise'
    train_data, test_data = get_test_data(gt_texts_path, noise_texts_path, size=500)

    # base_candidators = [HunspellCandidator(), SymSpellCandidator(2), SymSpellCandidator(3), NNCandidator(10)]
    base_candidators = [HunspellCandidator(), NNCandidator(num_beams=3), NNCandidator(num_beams=5), SymSpellCandidator(max_dictionary_edit_distance=2, prefix_length=7, count_threshold=1), SymSpellCandidator(max_dictionary_edit_distance=3, prefix_length=7, count_threshold=1)]
    stacked_candidators = [AggregatedCandidator([base_candidators[i] for i in range(0, k)]) for k in range(1, len(base_candidators) + 1)]

    # res = []
    # newly_solved_tasks = []
    # for stacked_candidator in stacked_candidators:
    #     found = 0
    #     not_found = 0
    #     total_mistakes = 0
    #     avg_cands = 0
    #     for spelled_text in tqdm(train_data):
    #         for spell in spelled_text.spells:
    #             total_mistakes += 1
    #             cands = stacked_candidator.get_candidates(spelled_text.text, [SpelledWord(spelled_text.text, (spell.start, spell.start + len(spell.spelled)))])
    #             if spell.correct in cands[0]:
    #                 found += 1
    #             avg_cands += len(cands[0])
    #     avg_cands /= total_mistakes
    #     avg_cands = round(avg_cands, 1)
    #     print(str(stacked_candidator), '\nTotal mistakes:', total_mistakes, 'Found:', "{0:.0%}".format(found/total_mistakes), 'Not_found:', "{0:.0%}".format((total_mistakes - found)/total_mistakes), 'Avg cands:', avg_cands)

    total_mistakes = 0
    res = {}
    for spelled_text in tqdm(train_data):
        for spell in spelled_text.spells:
            total_mistakes += 1
            candidator_ind = 0
            while candidator_ind < len(stacked_candidators):
                cands = stacked_candidators[candidator_ind].get_candidates(spelled_text.text, [
                    SpelledWord(spelled_text.text, (spell.start, spell.start + len(spell.spelled)))])[
                    0]

                if spell.correct in cands:
                    if not str(stacked_candidators[candidator_ind]) in res:
                        res[str(stacked_candidators[candidator_ind])] = []
                    res[str(stacked_candidators[candidator_ind])].append({'Spelled': spell.spelled, 'Correct': spell.correct})
                    break
                else:
                    candidator_ind += 1

            if candidator_ind == len(stacked_candidators):
                if not 'Absolutely not found' in res:
                    res['Absolutely not found'] = []
                res['Absolutely not found'].append({'Spelled': spell.spelled, 'Correct': spell.correct})

    print(res)

    for key in res:
        print(key, 'Gained ', "{0:.0%}".format(len(res[key])/total_mistakes))




if __name__ == '__main__':
    main()
