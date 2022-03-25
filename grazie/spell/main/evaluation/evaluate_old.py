import string
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from functools import cmp_to_key
from grazie.common.main.ranking.catboost_ranker import CatBoostRanker
from grazie.spell.main.data.base import SpelledText
from grazie.spell.main.data.utils import get_test_data, default_args_parser
from grazie.spell.main.model.candidator import AggregatedCandidator, BaseCandidator, IdealCandidator, LevenshteinCandidator, HunspellCandidator
from grazie.spell.main.model.detector import BaseDetector, DictionaryDetector, IdealDetector, HunspellDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector
from grazie.spell.main.model.ranker import SpellRanker, FeaturesSpellRanker, RandomSpellRanker
from grazie.spell.main.model.spellcheck_model import SpellCheckModelBase, SpellCheckModel
from grazie.spell.main.model.spellcheck_model import SpellCheckModel, SpellCheckModelCharBasedTransformerMedium
import copy


def remove_specific_metrics(model: SpellCheckModelBase, metric_values: Dict[str, float],
                            detector_metrics: List[str], candidator_metrics: List[str],
                            ranker_metrics: List[str]) -> Dict[str, float]:
    metrics_to_remove = []
    if isinstance(model, SpellCheckModel):
        if isinstance(model.detector, IdealDetector):
            metrics_to_remove.extend(detector_metrics)
        if isinstance(model.candidator, IdealCandidator):
            metrics_to_remove.extend(candidator_metrics)
        if isinstance(model.ranker, RandomSpellRanker):
            metrics_to_remove.extend(ranker_metrics)

    for metric_name in metrics_to_remove:
        del metric_values[metric_name]

    return metric_values


def evaluate(model: SpellCheckModelBase, data: List[SpelledText], verbose: bool = False, max_mistakes_log: int = 100, path_save_result: str = 'junk_result.txt'):

    metric_values: Dict[str, Any] = {"texts_num": len(data)}

    detector_matches = 0
    detector_precision_denom = 0
    detector_recall_denom = 0

    matched_positions = []

    mistake_not_found = []
    false_detection = []
    correct_cand_not_found = []
    ranking_mistake = []

    TP, FP, TN, FN_1, FN_2 = 0, 0, 0, 0, 0
    # P = TP / (TP + FP + FN_2)
    # R = TP / (TP + FN_1 + FN_2)

    # сюда сложим текст после исправления
    rewriting_result = []
    mistakes_found = []
    true_corrs = []

    # сюда gt тексты
    gt_texts = []

    # for TN calculation
    ttl_words = 0

    # correcting model spellings
    for spell_text in tqdm(data):
        text = spell_text.text
        ttl_words += len(text.split(' '))
        spells = spell_text.spells
        spell_results = model.check(text, true_spells=spells) # List[SpellCheckResult]

        # class SpellCheckResult:
        #     start: int
        #     finish: int
        #     variants: List[SpellCheckVariant]

        # class SpellCheckVariant:
        #     substitution: str
        #     score: float
        #     absolutely_best: bool = False
        new_spell_results = []
        for ind, pred_spell in enumerate(spell_results):
            real_l, real_r = pred_spell.start, pred_spell.finish
            while real_l > 0 and text[real_l - 1] != ' ':
                real_l -= 1
            while real_r < len(text) and text[real_r] != ' ':
                real_r += 1

            new_variants = []
            if len(pred_spell.variants) == 0:
                continue
            for pos, variant in enumerate(pred_spell.variants):
                variant.substitution = text[real_l: pred_spell.start] + variant.substitution + text[pred_spell.finish: real_r]
                new_variants.append(variant)
            pred_spell.variants = new_variants

            pred_spell.start = real_l
            pred_spell.finish = real_r
            new_spell_results.append(pred_spell)
        spell_results = new_spell_results

        # сразу сохраним исправленную версию текста
        shift = 0
        corrected_text = copy.copy(text)
        for pred_spell in spell_results:
            if len(pred_spell.variants) == 0:
                continue
            pref = corrected_text[: shift + pred_spell.start]
            suff = corrected_text[shift + pred_spell.finish:]
            corrected_text = pref + pred_spell.variants[0].substitution + suff
            shift += len(pred_spell.variants[0].substitution) - (pred_spell.finish - pred_spell.start)
        mistakes_found.append(len(spell_results))
        rewriting_result.append(corrected_text)

        # сразу сохраним GT версию текста
        gt_text = copy.copy(text)
        shift = 0
        for true_spell in spells:
            gt_text = gt_text[: shift + true_spell.start] + true_spell.correct + gt_text[shift + true_spell.start + len(true_spell.spelled):]
            shift += len(true_spell.correct) - len(true_spell.spelled)
        gt_texts.append(gt_text)

        detector_precision_denom += len(spell_results) # общее число исправлений
        detector_recall_denom += len(spells) # общее число ошибок

        not_found_spells = []

        cur_mistakes_corrected = 0

        # идем по всем GT опечаткам
        # Calculating TP, FN_1, FN_2
        for true_spell in spells:

            found = False
            for pred_spell in spell_results:
                real_l, real_r = pred_spell.start, pred_spell.finish
                while real_l > 0 and text[real_l - 1] != ' ':
                    real_l -= 1
                while real_r < len(text) and text[real_r] != ' ':
                    real_r += 1
                if true_spell.start == real_l and text[real_l: real_r] == true_spell.spelled:
                    found = True
                    detector_matches += 1
                    matched_position = float("inf")

                    correct_cand_found = False
                    for pos, variant in enumerate(pred_spell.variants):
                        real_cand = text[real_l: pred_spell.start] + variant.substitution + text[pred_spell.finish: real_r]

                        if real_cand == true_spell.correct:
                            matched_position = pos + 1
                            correct_cand_found = True
                            if pos > 0:
                                ranking_mistake.append(
                                    {'Bad Ratio': round(variant.score / pred_spell.variants[0].score, 2), 'Text': text,
                                     'Incorrect Word': true_spell.spelled, 'Corrected Word': true_spell.correct,
                                     'Candidates': [{'Word': variant.substitution, 'Score': round(variant.score, 2)} for
                                                    variant in pred_spell.variants[:5]]})
                            break
                    if not correct_cand_found:
                        correct_cand_not_found.append(
                            {'Text': text, 'Incorrect Word': true_spell.spelled, 'Corrected Word': true_spell.correct,
                             'Candidates': [{'Word': variant.substitution, 'Score': round(variant.score, 2)} for variant
                                            in pred_spell.variants[:5]]})
                    matched_positions.append(matched_position)

            if not found:
                mistake_not_found.append(
                    {'Text': text, 'Incorrect Word': true_spell.spelled, 'Corrected Word': true_spell.correct})
                not_found_spells.append(true_spell)
                FN_1 += 1
            else:
                if matched_positions[-1] > 1:
                    FN_2 += 1
                else:
                    TP += 1
                    cur_mistakes_corrected += 1

        true_corrs.append(cur_mistakes_corrected)


        # Calculating FP
        # идем по всем НЕ GT опечаткам и смотрим на false positives
        for pred_spell in spell_results:
            real = False
            for true_spell in spells:
                if true_spell.start == pred_spell.start and true_spell.spelled == text[
                                                                                  pred_spell.start:pred_spell.finish]:
                    real = True
                    break
            if real:
                continue
            FP += 1
            false_detection.append({'Text': text, 'Fake Incorrect Word': text[pred_spell.start: pred_spell.finish], 'Candidates': [{'Word': variant.substitution, 'Score': round(variant.score, 2)} for variant in pred_spell.variants[:3]]})

    # saving correction results
    with open(path_save_result, 'w') as f:
        for sent, spell_text, gt_text, mist_found, true_corr in zip(rewriting_result, data, gt_texts, mistakes_found, true_corrs):
            text = spell_text.text
            f.write(f'Number of mistakes: {len(spell_text.spells)}, Number of corrections: {mist_found}, Right corrections: {true_corr}\n{text} - Noised\n{gt_text} - GT\n{sent} - Corrected\n\n')

    TN = ttl_words - (TP + FP + FN_1 + FN_2)

    P = TP / (TP + FP + FN_2)
    R = TP / (TP + FN_1 + FN_2)
    word_level_accuracy = (TP + TN) / (TP + TN + FP + FN_1 + FN_2)

    beta = 0.5
    F_beta = (1 + beta ** 2) * P * R / ((P * beta ** 2) + R)


    metric_values["spells_num"] = detector_recall_denom
    metric_values["detector_precision"] = round(detector_matches / detector_precision_denom, 2)
    metric_values["detector_recall"] = round(detector_matches / detector_recall_denom, 2)

    metric_values['Precision'] = round(P, 2)
    metric_values['Recall'] = round(R, 2)
    metric_values['F_0.5'] = round(F_beta, 2)
    metric_values['Word-level accuracy'] = round(word_level_accuracy, 2)

    mistakes_examples = dict()
    mistakes_examples["mistake_not_found"] = mistake_not_found[: max_mistakes_log]
    mistakes_examples["false_detection"] = false_detection[: max_mistakes_log]
    mistakes_examples["correct_cand_not_found"] = correct_cand_not_found[: max_mistakes_log]

    def compare(item1, item2):
        if item1['Bad Ratio'] < item2['Bad Ratio']:
            return -1
        elif item1['Bad Ratio'] > item2['Bad Ratio']:
            return 1
        else:
            return 0
    ranking_mistake = sorted(ranking_mistake, key=cmp_to_key(compare))

    mistakes_examples["ranking_mistake"] = ranking_mistake[: max_mistakes_log]

    metric_values[f"candidator_acc (acc@inf)"] = round(float(np.mean([pos < float("inf") for pos in matched_positions])), 2)

    for k in [1, 3]:
        accuracy_k = float(np.mean([pos <= k for pos in matched_positions] + [False for i in range(len(mistake_not_found))]))
        metric_values[f"pipeline_acc@{k}"] = round(accuracy_k, 2)

    ranker_matched_positions = []
    for pos in matched_positions:
        if pos < float("inf"):
            ranker_matched_positions.append(pos)

    for k in [1, 3]:
        accuracy_k = float(np.mean([pos <= k for pos in ranker_matched_positions]))
        metric_values[f"ranker_acc@{k}"] = round(accuracy_k, 2)

    remove_specific_metrics(model, metric_values, ["detector_precision", "detector_recall"],
                            ["candidator_acc (acc@inf)"], ["acc@1", "acc@3", "mrr"])

    if verbose:
        for name, value in metric_values.items():
            print(f"{name}: {value}")

    # print('Not found spells:\n', not_found_spells)
    return metric_values, mistakes_examples


def evaluate_detector(detector: BaseDetector, data: List[SpelledText], verbose: bool = False):
    model = SpellCheckModel(detector, IdealCandidator(), RandomSpellRanker())
    metric_values = evaluate(model, data, verbose)
    return metric_values


def evaluate_candidator(candidator: BaseCandidator, data: List[SpelledText], verbose: bool = False):
    model = SpellCheckModel(IdealDetector(), candidator, RandomSpellRanker())
    metric_values = evaluate(model, data, verbose)
    return metric_values


def evaluate_ranker(ranker: SpellRanker, data: List[SpelledText], candidator: BaseCandidator, verbose: bool = False):
    model = SpellCheckModel(IdealDetector(), AggregatedCandidator([IdealCandidator(), candidator]), ranker)
    metric_values = evaluate(model, data, verbose)
    return metric_values


def run_detector():
    args = default_args_parser().parse_args()

    _, test_data = get_test_data(args.texts_path, args.size)

    detector = IdealDetector()
    # detector_precision: 1.0
    # detector_recall: 1.0

    detector = DictionaryDetector()
    # detector_precision: 0.7971014492753623
    # detector_recall: 0.8870967741935484

    detector = HunspellDetector()
    # detector_precision: 0.9017857142857143
    # detector_recall: 0.8782608695652174
    evaluate_detector(detector, test_data, verbose=True)


def run_candidator():
    args = default_args_parser().parse_args()

    _, test_data = get_test_data(args.texts_path, args.size)

    candidator = IdealCandidator()
    # candidator_accuracy: 1.0

    candidator = LevenshteinCandidator(max_err=2, index_prefix_len=2)
    # candidator_accuracy: 0.39473684210526316

    candidator = LevenshteinCandidator(max_err=2, index_prefix_len=1)
    # candidator_accuracy: 0.6545454545454545

    candidator = HunspellCandidator()
    # candidator_accuracy: 0.8769230769230769

    candidator = AggregatedCandidator([HunspellCandidator(), LevenshteinCandidator(max_err=2, index_prefix_len=1)])
    # candidator_accuracy: 0.8878504672897196

    evaluate_candidator(candidator, test_data, verbose=True)


def run_ranker():
    parser = default_args_parser()
    parser.add_argument("--freqs_path", type=str, required=True, help="Path to frequencies dictionary")
    parser.add_argument("--ranker_path", type=str, required=True, help="Saved CatBoost ranker")
    args = parser.parse_args()

    _, test_data = get_test_data(args.texts_path, args.size)

    features_collector = FeaturesCollector(args.freqs_path)
    rank_model = CatBoostRanker().load(args.ranker_path)
    ranker = FeaturesSpellRanker(features_collector, rank_model)

    evaluate_ranker(ranker, test_data, IdealCandidator(), verbose=True)


if __name__ == '__main__':
    # path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
    path_prefix = '/home/ubuntu/omelnikov/'
    test_gt = path_prefix + 'grazie/spell/main/data/datasets/test.bea500.norm'
    test_noise = path_prefix + 'grazie/spell/main/data/datasets/test.bea500.noise.norm'
    dataset_name = test_gt.split('/')[-1]
    test_data, test_data_fict = get_test_data(test_gt, test_noise, size=20, train_part=1.0)
    char_based_transformer = SpellCheckModelCharBasedTransformerMedium(
        checkpoint=path_prefix + 'grazie/spell/main/training/model_small_2_4.pt')
    pipeline_metrics, pipeline_mistakes = evaluate(char_based_transformer, test_data, verbose=True, max_mistakes_log=10)




    # run_detector()
    # run_candidator()
    # run_ranker()

# --texts_path ../../../nlc/main/evaluation/text.txt
# --freqs_path
