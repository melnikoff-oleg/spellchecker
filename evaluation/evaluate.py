import json
import datetime
import os
from tqdm import tqdm
from model.spellcheck_model import *
from data_utils.utils import get_texts_from_file
import string


# one can make saving to file through decorator

PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/grazie/spell/main/'


def evaluate(model: SpellCheckModelBase, texts_gt: List[str], texts_noise: List[str], exp_save_dir: str = None) -> Dict:
    tp, fp_1, fp_2, tn, fn = 0, 0, 0, 0, 0
    broken_tokenization_cases = 0
    fp_1_examples, fp_2_examples, fn_examples = [], [], []

    # Prepare folder and file to save info
    if exp_save_dir is not None:
        if not os.path.exists(exp_save_dir):
            os.makedirs(exp_save_dir)
        open(exp_save_dir + 'result.txt', 'w').close()

    # Iterating over all texts, comparing corrected version to gt
    for text_gt, text_noise in tqdm(zip(texts_gt, texts_noise), total=len(texts_gt)):
        text_res = model.correct(text_noise)
        words_gt, words_noise, words_res = text_gt.split(' '), text_noise.split(' '), text_res.split(' ')

        # If tokenization not preserved, then do nothing
        broken_tokenization = False
        if len(words_res) != len(words_noise):
            if exp_save_dir is not None:
                with open(exp_save_dir + 'result.txt', 'a+') as result_file:
                    result_file.write(f'Tokenization not preserved\n')
            broken_tokenization_cases += 1
            real_res = text_res
            text_res = text_noise
            words_res = words_noise
            broken_tokenization = True

        cur_tp, cur_fp_1, cur_fp_2, cur_tn, cur_fn = 0, 0, 0, 0, 0
        for idx, (word_gt, word_init) in enumerate(zip(words_gt, words_noise)):
            word_res = words_res[idx] if idx < len(words_res) else ''
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
        if exp_save_dir is not None:
            with open(exp_save_dir + 'result.txt', 'a+') as result_file:
                result_file.write(f'TP: {cur_tp}, FP_1: {cur_fp_1}, FP_2: {cur_fp_2}, FN: {cur_fn}, TN: {cur_tn}\n')

        # Updating global tp, fp, ...
        tp, fp_1, fp_2, tn, fn = tp + cur_tp, fp_1 + cur_fp_1, fp_2 + cur_fp_2, tn + cur_tn, fn + cur_fn

        # Return text_res to real value for writing to file
        if broken_tokenization:
            text_res = real_res

        # Writing correction results for current text
        if exp_save_dir is not None:
            with open(exp_save_dir + 'result.txt', 'a+') as result_file:
                result_file.write(f'{text_noise} - Noised\n{text_gt} - GT\n{text_res} - Result\n\n')

    # Calculating metrics
    word_level_accuracy = round((tp + tn) / (tp + fp_1 + fp_2 + tn + fn), 2)
    precision = round(tp / (tp + fp_1 + fp_2), 2) if (tp + fp_1 + fp_2) > 0 else 0
    recall = round(tp / (tp + fn), 2)
    f_0_5 = round((1 + 0.5 ** 2) * precision * recall / ((precision * 0.5 ** 2) + recall), 2) \
        if (precision > 0 or recall > 0) else 0
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
    if exp_save_dir is not None:
        with open(exp_save_dir + 'report.json', 'w') as result_file:
            json.dump(report, result_file, indent=4)

    # Printing report
    print(f'\nEvaluation metrics:\n\n{report["Metrics"]}')

    return report


def evaluate_ranker(model: DetectorCandidatorRanker, texts_gt: List[str], texts_noise: List[str], exp_save_dir: str = None) -> Dict:
    t, f = 0, 0
    f_examples = []

    # Prepare folder and file to save info
    if exp_save_dir is not None:
        if not os.path.exists(exp_save_dir):
            os.makedirs(exp_save_dir)

    def intersect_segments(l1, r1, l2, r2):
        return max(0, min(r1, r2) - max(l1, l2))

    # Iterating over all texts, comparing corrected version to gt
    for text_gt, text_noise in tqdm(zip(texts_gt, texts_noise), total=len(texts_gt)):
        text_res, spelled_words, candidates, corrections = model.correct(text_noise, return_all_stages=True)
        words_gt, words_noise, words_res = text_gt.split(' '), text_noise.split(' '), text_res.split(' ')

        # If tokenization not preserved, then continue
        if len(words_res) != len(words_noise):
            continue

        cur_ind = 0
        for word_gt, word_init, word_res in zip(words_gt, words_noise, words_res):
            cur_end = cur_ind + len(word_init)
            if word_init != word_gt:
                if word_res == word_gt:
                    t += 1
                else:
                    spelled_word_index = None
                    for idx, spelled_word in enumerate(spelled_words):
                        if intersect_segments(cur_ind, cur_end, spelled_word.interval[0], spelled_word.interval[1]) >= len(word_init) - 2:
                            spelled_word_index = idx
                            break
                    if spelled_word_index is None:
                        continue
                    cur_candidates = candidates[spelled_word_index]
                    correct_variant_exists = False
                    for cand in cur_candidates:
                        x = [cand, word_gt]
                        for i in range(2):
                            x[i] = x[i].strip()
                            x[i] = x[i].translate(str.maketrans('', '', string.punctuation))
                        if x[0] == x[1]:
                            correct_variant_exists = True
                            break
                    if correct_variant_exists:
                        f += 1
                        f_examples.append([f'Sentence: {text_noise}, Corr word: {word_gt}, Res word: {word_res}'])

            cur_ind += 1 + len(word_init)

    # Calculating metrics
    precision_at_1 = round(t / (t + f), 2)

    # Leave at most 3 random examples of each mistake
    sample = lambda array: random.sample(array, min(len(array), 10))
    f_examples = sample(f_examples)

    # Collecting all evaluation info to one json
    report = {
        'Date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        'Model': str(model),
        'Metrics': {
            'Precision@1': precision_at_1,
            'True': t,
            'False': f
        },
        'Mistakes examples': {
            'Wrong candidate choosen': f_examples,
        }
    }

    # Saving evaluation report
    if exp_save_dir is not None:
        with open(exp_save_dir + 'report.json', 'w') as result_file:
            json.dump(report, result_file, indent=4)

    # Printing report
    print(f'\nEvaluation report:\n{report}\n')

    return report


def evaluation_test():
    d_model = 256
    checkpoint = 'training/model_big_0_9.pt'
    model = CharBasedTransformerChecker(config={'d_model': d_model, 'encoder_layers': 6, 'decoder_layers': 6,
                                         'encoder_attention_heads': 8, 'decoder_attention_heads': 8,
                                         'encoder_ffn_dim': d_model * 4, 'decoder_ffn_dim': d_model * 4},
                                 checkpoint=PATH_PREFIX + checkpoint)
    texts_gt, texts_noise = get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea50.gt'), \
                            get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea50.noise')

    evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/char_based_transformer_big_10_epochs_test/')


if __name__ == '__main__':
    texts_gt, texts_noise = get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea10.gt'), \
                            get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea10.noise')

    # neuspell
    # model = SpellCheckModelNeuSpell()
    # evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/neuspell-bert/')


    # FST
    # model = FastProdModel()
    # evaluate_ranker(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/fst-ranker/')


    # distil bart de05
    # checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent-distil-dec05_v0_81396.pt'
    # config = BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
    #                                         encoder_attention_heads=12, decoder_layers=3, decoder_ffn_dim=3072,
    #                                         decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
    #                                         activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
    #                                         activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
    #                                         use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
    #                                         is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2)
    # device = torch.device('cuda')
    # model = BartForConditionalGeneration(config)
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.save_pretrained(PATH_PREFIX + 'training/checkpoints/distilbart-best')
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # tokenizer.save_pretrained(PATH_PREFIX + 'training/checkpoints/distilbart-best')
    # tokenizer.push_to_hub("distilbart-sep-mask-all")
    # model = model.to(device)
    # model.push_to_hub("distilbart-sep-mask-all")
    # model = BartSepMaskAllChecker(model=model)
    checker = DCR()
    checker.from_pretrained()
    evaluate(checker, texts_gt, texts_noise,
             PATH_PREFIX + 'experiments/distilbart-sepmaskall0-de05-BEA500/')


    # detector candidator ranker
    # model = DetectorCandidatorRanker()
    # evaluate_ranker(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/3-stage-oldbart-ranker/')

    # bert bart 214056 1236504
    # model_name = 'bart-sep-mask-all-sent_v0_214056'
    # checkpoint = f'training/checkpoints/{model_name}'
    # model = BertBartChecker(checkpoint=PATH_PREFIX + checkpoint + '.pt', device=torch.device('cuda'))
    # evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/bart-sep-mask-all-BEA60k/')

    # detector candidator ranker
    # model = DetectorCandidatorRanker()
    # evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/3-stage-oldbartLN+lev/')

    # bart-base
    # checkpoint = 'training/checkpoints/bart-base_v1_4.pt'
    # model = BartChecker(checkpoint=PATH_PREFIX + checkpoint, device=torch.device('cuda'))
    # evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/bart-base/')

    # char-based
    # checkpoint = 'training/checkpoints/char-based-xl-explode_v1_9.pt'
    # tokenizer = CharBasedTransformerChecker.BartTokenizer(
    #     PATH_PREFIX + 'data_utils/char_based_transformer_vocab/url_vocab.json',
    #     PATH_PREFIX + 'data_utils/char_based_transformer_vocab/url_merges.txt'
    # )
    # d_model = 512
    # config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=d_model, encoder_layers=6, decoder_layers=6,
    #                     encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=d_model * 4,
    #                     decoder_ffn_dim=d_model * 4)
    # model = CharBasedTransformerChecker(checkpoint=PATH_PREFIX + checkpoint, device=torch.device('cuda:0'),
    #                                     config=config)
    # evaluate(model, texts_gt, texts_noise, PATH_PREFIX + 'experiments/char-level-bea50/')

    # with open(PATH_PREFIX + 'dataset/1blm/1blm.train.noise.sep_mask_all_sent') as f:
    #     for line in f:
    #         print(line)
    #         break
