import json

from grazie.spell.main.data.utils import get_texts_from_file
from grazie.spell.main.model.spellcheck_model_best import SpellCheckModelBase, CharBasedTransformer
from typing import List
import random
import datetime
import os
from tqdm import tqdm


def evaluate(model: SpellCheckModelBase, texts_gt: List[str], texts_noise: List[str], exp_save_dir: str):
    tp, fp_1, fp_2, tn, fn = 0, 0, 0, 0, 0
    broken_tokenization_cases = 0
    fp_1_examples, fp_2_examples, fn_examples = [], [], []

    # Prepare folder and file to save info
    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir)
    open(exp_save_dir + 'result.txt', 'w').close()

    # Iterating over all texts, comparing corrected version to gt
    for text_gt, text_noise in tqdm(zip(texts_gt, texts_noise), total=len(texts_gt)):
        text_res = model.correct(text_noise)
        words_gt, words_noise, words_res = text_gt.split(' '), text_noise.split(' '), text_res.split(' ')

        # Dumb check if tokenization not preserved
        if len(words_res) != len(words_noise):
            broken_tokenization_cases += 1
            # Writing info for current text
            with open(exp_save_dir + 'result.txt', 'a+') as result_file:
                result_file.write(f'Tokenization not preserved\n')
        else:
            cur_tp, cur_fp_1, cur_fp_2, cur_tn, cur_fn = 0, 0, 0, 0, 0
            for word_gt, word_init, word_res in zip(words_gt, words_noise, words_res):
                word_report = {'Text noise': text_noise, 'Word noise': word_init, 'Word gt': word_gt, 'Word res': word_res}
                if word_init == word_gt:
                    if word_res == word_gt:
                        cur_tn += 1
                    else:
                        cur_fp_1 += 1
                        fp_1_examples.append(word_report)
                else:
                    if word_res == word_gt:
                        cur_tp += 1
                    else:
                        if word_res == word_init:
                            cur_fn += 1
                            fn_examples.append(word_report)
                        else:
                            cur_fp_2 += 1
                            fp_2_examples.append(word_report)

            # Writing info for current text
            with open(exp_save_dir + 'result.txt', 'a+') as result_file:
                result_file.write(
                    f'TP: {cur_tp}, FP_1: {cur_fp_1}, FP_2: {cur_fp_2}, FN: {cur_fn}, TN: {cur_tn}\n')

            # Updating global tp, fp, ...
            tp, fp_1, fp_2, tn, fn = tp + cur_tp, fp_1 + cur_fp_1, fp_2 + cur_fp_2, tn + cur_tn, fn + cur_fn

        # Writing correction results for current text
        with open(exp_save_dir + 'result.txt', 'a+') as result_file:
            result_file.write(f'{text_noise} - Noised\n{text_gt} - GT\n{text_res} - Result\n\n')

    # Calculating metrics
    word_level_accuracy = round((tp + tn) / (tp + fp_1 + fp_2 + tn + fn), 2)
    precision = round(tp / (tp + fp_1 + fp_2), 2)
    recall = round(tp / (tp + fn), 2)
    f_0_5 = round((1 + 0.5 ** 2) * precision * recall / ((precision * 0.5 ** 2) + recall), 2)
    broken_tokenization_cases = round(broken_tokenization_cases / len(texts_gt), 2)

    # Leave at most 3 random examples of each mistake
    sample = lambda array: random.sample(array, min(len(array), 3))
    fp_1_examples, fp_2_examples, fn_examples = sample(fp_1_examples), sample(fp_2_examples), sample(fn_examples)

    # Collecting all evaluation info to one json
    report = {
        'Date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        'Model': str(model),
        'Metrics': {
            'Precision': precision,
            'Recall': recall,
            'F_0_5': f_0_5,
            'Word-level accuracy': word_level_accuracy,
            'Broken tokenization cases': broken_tokenization_cases,
        },
        'Mistakes examples': {
            'Wrong correction of real mistake': fp_2_examples,
            'No mistake, but model corrected': fp_1_examples,
            'Not found mistake': fn_examples
        }
    }

    # Saving evaluation report
    with open(exp_save_dir + 'report.json', 'w') as result_file:
        json.dump(report, result_file, indent=4)

    # Printing report
    print(f'\nEvaluation report:\n\n{report}')


def evaluation_test():
    path_prefix = '/home/ubuntu/omelnikov/grazie/spell/main/'
    model = CharBasedTransformer(config={'d_model': 256, 'encoder_layers': 6, 'decoder_layers': 6,
                                         'encoder_attention_heads': 8, 'decoder_attention_heads': 8,
                                         'encoder_ffn_dim': 1024, 'decoder_ffn_dim': 1024},
                                 checkpoint=path_prefix + 'training/model_big_0_9.pt')
    texts_gt, texts_noise = get_texts_from_file(path_prefix + 'data/datasets/bea/bea10.gt'), \
                            get_texts_from_file(path_prefix + 'data/datasets/bea/bea10.noise')

    evaluate(model, texts_gt, texts_noise, path_prefix + 'data/experiments/char_based_transformer_big_9_epochs/')


if __name__ == '__main__':
    evaluation_test()
