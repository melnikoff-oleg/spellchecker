from abc import ABC, abstractmethod
from model.base import SpelledWord
from typing import List, Dict
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from model.candidator import HunspellCandidator
from model.detector import HunspellDetector
import math
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


class BaseRanker(ABC):
    @abstractmethod
    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        raise NotImplementedError


def test_tokenization():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)

    text = 'For the activities, I chose climbing because I took a cours for 2 weeks last year and now I have a good ' \
           'level of proficiency. For the other one I chose photography but I am not a proffesional, I just take ' \
           'some pictures on my holiday!'

    text_start = 'For the activities, I chose climbing because I took a '

    start_tokens_enc = tokenizer(text_start[:-1])
    print('Tokenized start')
    print(tokenizer.convert_ids_to_tokens(start_tokens_enc["input_ids"]))
    print('Tokenized word')
    word_tokens_enc = tokenizer('course', add_special_tokens=False)
    print(tokenizer.convert_ids_to_tokens(word_tokens_enc["input_ids"]))

    # cands_ranges += [(len(tokenizer.encode(text_start[:-1])),
    #                   len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]


class BartSepMaskAllRanker(BaseRanker):

    def __init__(self, checkpoint_path: str = '----', config=None, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config is None:
            config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
        self.model = BartForConditionalGeneration(config)
        # Model was trained on GPU, maybe we are inferring on CPU
        if self.device == torch.device('cpu'):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = self.model.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model.eval()

    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        all_candidates = []
        for i, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            text, word, start, finish = spelled_word.text, spelled_word.word, spelled_word.interval[0], spelled_word.interval[1]
            text_pref = text[: start]
            text_suff = text[finish:]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                all_candidates += cands
                texts_inds += [i for _ in cands]

                sep_token = '</s>'
                # sep_token = '<sep>'

                input_text = word + f' {sep_token} ' + text_pref + '<mask>' + text_suff
                texts += [input_text for _ in cands]

                output_texts = [text_pref + syn + text_suff for syn in cands]
                outs += output_texts

                cands_ranges += [(len(self.tokenizer.encode(text_pref[:-1])),
                                  len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]

        # DEBUG
        # print(f'Input BART texts: {texts}')
        # print()
        # print(f'Output BART texts: {outs}')

        encoded_input = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True, padding=True
        ).to(self.device)['input_ids']
        encoded_output = self.tokenizer.batch_encode_plus(
            outs,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True, padding=True
        ).to(self.device)['input_ids']

        all_logits = self.model(encoded_input, labels=encoded_output).logits.cpu()

        scores: Dict[int, List[float]] = {}
        for i, logits in enumerate(all_logits):

            ind = texts_inds[i]
            if ind not in scores:
                scores[ind] = []

            # DEBUG print all tokens and their probs
            # if 'course' in outs[i] or 'ours' in outs[i]:
            #     tokenized_input = self.tokenizer(outs[i])
            #     tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
            #     print(f'Output tokens:{tokens}')
            #     token_probs = []
            #     all_probs = torch.softmax(logits, dim=1)
            #     for j, token_idx in enumerate(encoded_output[i]):
            #         token_probs.append(all_probs[j, token_idx].item())
            #     print(f'Output tokens probs: {token_probs}')

            syn_range = cands_ranges[i]

            word_logits = logits[syn_range[0] - 1:syn_range[0] + syn_range[1] - 1]
            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][syn_range[0] - 1:syn_range[0] + syn_range[1] - 1]):
                word_log_prob += log_probs[j, token_idx]

            scores[ind].append(math.exp(word_log_prob.item()))

        result: List[str] = ['' for _ in spelled_words]
        for i in scores:
            mx = -1e18
            mx_ind = None
            for j, score in enumerate(scores[i]):
                # DEBUG
                print(f'Candidate: {candidates[i][j]}, score: {score}')
                if mx < score:
                    mx = score
                    mx_ind = j
            result[i] = candidates[i][mx_ind]
        return result
# Sep Mask All


class BartFineTuneRanker(BaseRanker):
    def __init__(self, checkpoint_path: str = PATH_PREFIX + 'training/checkpoints/bart-base_v1_4.pt', device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
        self.model = BartForConditionalGeneration(config)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = self.model.to(device)
        self.model.eval()

    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        all_candidates = []
        for i, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            text, start, finish = spelled_word.text, spelled_word.interval[0], spelled_word.interval[1]
            text_start = text[: start]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                all_candidates += cands
                texts_inds += [i for _ in cands]

                # input_text = text
                texts += [text for _ in cands]

                output_texts = [text_start + syn + text[finish:] for syn in cands]
                outs += output_texts

                cands_ranges += [(len(self.tokenizer.encode(text_start[:-1])),
                                  len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]

        encoded_input = self.tokenizer(texts, return_tensors='pt', truncation=True,
                                       padding=True).to(self.device)['input_ids']
        encoded_output = self.tokenizer(outs, return_tensors='pt', truncation=True,
                                        padding=True).to(self.device)['input_ids']

        all_logits = self.model(encoded_input, labels=encoded_output).logits.cpu()


        scores: Dict[int, List[float]] = {}
        for i, logits in enumerate(all_logits):

            ind = texts_inds[i]
            if ind not in scores:
                scores[ind] = []

            syn_range = cands_ranges[i]

            word_logits = logits[syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]

            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]):
                word_log_prob += log_probs[j, token_idx]

            scores[ind].append(math.exp(word_log_prob.item()))

        result: List[str] = ['' for _ in spelled_words]
        for i in scores:
            mx = -1e18
            mx_ind = None
            for j, score in enumerate(scores[i]):
                if mx < score:
                    mx = score
                    mx_ind = j
            result[i] = candidates[i][mx_ind]
        return result


class BartRanker(BaseRanker):
    def __init__(self, checkpoint_path: str = 'facebook/bart-base', device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
        self.model.eval()

    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        all_candidates = []
        for i, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            text, start, finish = spelled_word.text, spelled_word.interval[0], spelled_word.interval[1]
            text_start = text[: start]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                all_candidates += cands
                texts_inds += [i for _ in cands]

                input_text = text[:start] + '<mask>' + text[finish:]
                texts += [input_text for _ in cands]

                output_texts = [text_start + syn + text[finish:] for syn in cands]
                outs += output_texts

                cands_ranges += [(len(self.tokenizer.encode(text_start[:-1])),
                                  len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]

        # DEBUG
        # print(f'Input BART texts: {texts}')
        # print()
        # print(f'Output BART texts: {outs}')

        encoded_input = self.tokenizer(texts, return_tensors='pt', truncation=True,
                                       padding=True).to(self.device)['input_ids']
        encoded_output = self.tokenizer(outs, return_tensors='pt', truncation=True,
                                        padding=True).to(self.device)['input_ids']

        all_logits = self.model(encoded_input, labels=encoded_output).logits.cpu()

        scores: Dict[int, List[float]] = {}
        for i, logits in enumerate(all_logits):

            ind = texts_inds[i]
            if ind not in scores:
                scores[ind] = []

            # DEBUG print all tokens and their probs
            # if ' course' in outs[i] or ' cause' in outs[i]:
            #     # tokenized_input = self.tokenizer(outs[i])
            #     tokens = self.tokenizer.convert_ids_to_tokens(encoded_output[i])
            #     # print(f'Output tokens:{tokens}')
            #     token_probs = []
            #     all_probs = torch.softmax(logits, dim=1)
            #     for j, token_idx in enumerate(encoded_output[i]):
            #         token_probs.append(all_probs[j, token_idx].item())
            #     for tok, pro in zip(tokens, token_probs):
            #         print(f'Token: {tok}, Prob: {pro}')
                # print(f'Output tokens probs: {token_probs}')

            syn_range = cands_ranges[i]

            word_logits = logits[syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]
            # if ' course ' in outs[i] or ' court ' in outs[i]:
            #     print('Word logits:', word_logits)
            #     print()
            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]):
                word_log_prob += log_probs[j, token_idx]

            scores[ind].append(math.exp(word_log_prob.item()))

        result: List[str] = ['' for _ in spelled_words]
        for i in scores:
            mx = -1e18
            mx_ind = None
            for j, score in enumerate(scores[i]):
                # DEBUG
                # print(f'Candidate: {candidates[i][j]}, score: {score}')
                if mx < score:
                    mx = score
                    mx_ind = j
            result[i] = candidates[i][mx_ind]
        return result



def test_bart():

    # PICK CHECKPOINT
    checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-base_v1_4.pt'
    # checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask_v1_3.pt'
    # checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent-distil-dec05_v0_248088.pt'
    # checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent_v0_1236504.pt'

    # PICK CONFIG
    config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
    # config = BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
    #                                         encoder_attention_heads=12, decoder_layers=3, decoder_ffn_dim=3072,
    #                                         decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
    #                                         activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
    #                                         activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
    #                                         use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
    #                                         is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2)
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # INIT ALL
    device = torch.device('cuda')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # model = BartForConditionalGeneration(config)
    # model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    # CHOOSE TEXT TO TEST
    input_text = 'Of couse, the shopkeepers are human beings as well.'
    output_text = 'Of course, the shopkeepers are human beings as well.'

    input_ids = tokenizer([input_text], return_tensors='pt').to(device)['input_ids']
    output_ids = tokenizer([output_text], return_tensors='pt').to(device)['input_ids']
    logits = model(input_ids).logits

    for masked_index in range(logits.shape[1]):
        probs = logits[0, masked_index].softmax(dim=0)
        values, predictions = probs.topk(5)

        print(tokenizer.decode(predictions).split())

    print('Labels')

    logits = model(input_ids, labels=output_ids).logits

    for masked_index in range(logits.shape[1]):
        probs = logits[0, masked_index].softmax(dim=0)
        values, predictions = probs.topk(5)

        print(tokenizer.decode(predictions).split())

    print('Decoder Input Itds')

    logits = model(input_ids, decoder_input_ids=output_ids).logits

    for masked_index in range(logits.shape[1]):
        probs = logits[0, masked_index].softmax(dim=0)
        values, predictions = probs.topk(5)

        print(tokenizer.decode(predictions).split())


def test():
    # text = 'Harry warks in cofee company'
    # text = 'For the activities, I chose climbing because I took a cours for 2 weeks last year and now I have a good ' \
    #        'level of proficiency. For the other one I chose photography but I am not a proffesional, I just take ' \
    #        'some pictures on my holiday!'
    # text = 'Of couse, the shopkeepers are human beings as well.'
    text = 'In addition to this, you may have some roboters bringing the newspaper to the table, tidying up the house, and doing the shopping. Maybe they will also be your life patners.'
    spelled_words: List[SpelledWord] = HunspellDetector().detect(text)
    detections = [i.word for i in spelled_words]
    candidates = HunspellCandidator().get_candidates(text, spelled_words)
    # ranker = BartRanker(device=torch.device('cuda'))
    # config = BartConfig(vocab_size = 50265, max_position_embeddings = 1024, encoder_layers = 6, encoder_ffn_dim = 3072,
    #                     encoder_attention_heads = 12, decoder_layers = 3, decoder_ffn_dim = 3072,
    #                     decoder_attention_heads = 12, encoder_layerdrop = 0.0, decoder_layerdrop = 0.0,
    #                     activation_function = 'gelu', d_model = 768, dropout = 0.1, attention_dropout = 0.0,
    #                     activation_dropout = 0.0, init_std = 0.02, classifier_dropout = 0.0, scale_embedding = False,
    #                     use_cache = True, num_labels=3, pad_token_id = 1, bos_token_id = 0, eos_token_id = 2,
    #                     is_encoder_decoder = True, decoder_start_token_id = 2, forced_eos_token_id = 2)
    # ranker = BartSepMaskAllRanker(
    #     checkpoint_path=PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent-distil-dec05_v0_248088.pt',
    #     config=config,
    #     device=torch.device('cuda'))
    ranker = BartSepMaskAllRanker(
        checkpoint_path=PATH_PREFIX + 'training/checkpoints/bart-sep-mask_v1_3.pt',
        device=torch.device('cuda'))
    final_corrections = ranker.rank(text, spelled_words, candidates)
    print(f'Text: {text}\nHunspell detections: {detections}\nHunspell candidates: {candidates}\nBART ranker results: {final_corrections}\n')


if __name__ == '__main__':
    test_bart()
    # test()
    # test_tokenization()
