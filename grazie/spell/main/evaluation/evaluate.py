from typing import List, Dict, Any
import numpy as np
import pandas as pd
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


def evaluate(model: SpellCheckModelBase, data: List[SpelledText], verbose: bool = False, max_mistakes_log: int = 100):

    metric_values: Dict[str, Any] = {"texts_num": len(data)}

    detector_matches = 0
    detector_precision_denom = 0
    detector_recall_denom = 0

    matched_positions = []

    mistake_not_found = []
    false_detection = []
    correct_cand_not_found = []
    ranking_mistake = []

    for spell_text in tqdm(data):
        text = spell_text.text
        spells = spell_text.spells
        spell_results = model.check(text, true_spells=spells)

        detector_precision_denom += len(spell_results)
        detector_recall_denom += len(spells)

        not_found_spells = []

        # идем по всем GT опечаткам
        for true_spell in spells:

            found = False
            for pred_spell in spell_results:
                if true_spell.start == pred_spell.start and true_spell.spelled == text[
                                                                                  pred_spell.start:pred_spell.finish]:
                    found = True
                    detector_matches += 1
                    matched_position = float("inf")

                    correct_cand_found = False
                    for pos, variant in enumerate(pred_spell.variants):
                        if variant.substitution == true_spell.correct:
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

            false_detection.append({'Text': text, 'Fake Incorrect Word': text[pred_spell.start: pred_spell.finish], 'Candidates': [{'Word': variant.substitution, 'Score': round(variant.score, 2)} for variant in pred_spell.variants[:3]]})

    metric_values["spells_num"] = detector_recall_denom
    metric_values["detector_precision"] = round(detector_matches / detector_precision_denom, 2)
    metric_values["detector_recall"] = round(detector_matches / detector_recall_denom, 2)

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
    # run_detector()
    # run_candidator()
    run_ranker()

# --texts_path ../../../nlc/main/evaluation/text.txt
# --freqs_path
