from abc import abstractmethod, ABC
from typing import List, Optional

from dataclasses import dataclass

from grazie.spell.main.model.candidator import BaseCandidator, AggregatedCandidator
from grazie.spell.main.model.detector import BaseDetector
from grazie.spell.main.model.ranker import SpellRanker

# from happytransformer import HappyTextToText

# from happytransformer import TTSettings

import random

from transformers import RobertaTokenizer
import string
import json
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import BartConfig, BartForConditionalGeneration
from tqdm import tqdm
import timeit
from transformers import get_linear_schedule_with_warmup

import time

@dataclass
class SpellCheckVariant:
    substitution: str
    score: float
    absolutely_best: bool = False


@dataclass
class SpellCheckResult:
    start: int
    finish: int
    variants: List[SpellCheckVariant]


class SpellCheckModelBase(ABC):
    def __init__(self, max_count: int = 5):
        self.max_count = max_count

    @abstractmethod
    def check(self, text: str, max_count: Optional[int] = None, round_digits: int = None, **kwargs) -> List[SpellCheckResult]:
        raise NotImplementedError


class SpellCheckModelE2E(SpellCheckModelBase):

    @abstractmethod
    def correct_string(self, s) -> str:
        raise NotImplementedError

    def check(self, text: str, max_count: Optional[int] = None, round_digits: int = None, **kwargs) -> List[SpellCheckResult]:
        corrected_text = self.correct_string(text)
        ans = []
        ind = 0
        for i, j in zip(text.split(' '), corrected_text.split(' ')):
            if i != j:
                ans.append(SpellCheckResult(start=ind, finish=ind + len(i), variants=[SpellCheckVariant(substitution=j, score=1)]))
            ind += len(i) + 1

        return ans


class BartTokenizer(RobertaTokenizer):
    """
    Construct a BART tokenizer.
    [`BartTokenizer`] is identical to [`RobertaTokenizer`]. Refer to superclass [`RobertaTokenizer`] for usage examples
    and documentation concerning the initialization parameters and other methods.
    """
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}


def create_vocab_files():
    chars = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + list(string.punctuation) + list(string.digits) + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    url_vocab = {c: i for i, c in enumerate(chars)}
    with open("url_vocab.json", 'w') as json_file:
      json.dump(url_vocab, json_file)

    merges = "#version: 0.2\n"
    with open("url_merges.txt", 'w') as f:
        f.write(merges)


class SpellCheckModelCharBasedTransformerMedium(SpellCheckModelE2E):

    def __init__(self, max_count: Optional[int] = None, checkpoint: str = 'model_sch_lin_warm_239_2.pt'):
        super().__init__(max_count)
        self.checkpoint = checkpoint
        create_vocab_files()
        tokenizer = BartTokenizer("url_vocab.json", "url_merges.txt")
        config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=128, encoder_layers=6, decoder_layers=6,
                            encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=512,
                            decoder_ffn_dim=512)
        model = BartForConditionalGeneration(config)

        # model was on GPU, maybe we are infering on CPU
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint))
        else:
            model.load_state_dict(torch.load(checkpoint, map_location ='cpu'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

    def __str__(self):
        return 'Char-Based Transformer Medium: ' + str(self.model).split('(')[0] + ' ' + self.checkpoint.split('/')[-1]

    def correct_string(self, s) -> str:
        s = s.replace(' ', '_')
        ans_ids = self.model.generate(self.tokenizer([s], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=100)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ind, i in enumerate(ans_tokens):
            ans_tokens[ind] = ans_tokens[ind].replace('_', ' ')[7:].split('<')[0]

        return ' '.join(ans_tokens)


class SpellCheckModelCharBasedTransformerSmall(SpellCheckModelE2E):

    def __init__(self, max_count: Optional[int] = None, checkpoint: str = 'model_small_1_1.pt'):
        super().__init__(max_count)
        create_vocab_files()
        tokenizer = BartTokenizer("url_vocab.json", "url_merges.txt")
        config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=128, encoder_layers=2, decoder_layers=2,
                            encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=512,
                            decoder_ffn_dim=512)
        model = BartForConditionalGeneration(config)

        # model was on GPU, maybe we are infering on CPU
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint))
        else:
            model.load_state_dict(torch.load(checkpoint, map_location ='cpu'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

    def __str__(self):
        return 'Char-Based Transformer Small: ' + str(self.model).split('(')[0]

    def correct_string(self, s) -> str:
        s = s.replace(' ', '_')
        ans_ids = self.model.generate(self.tokenizer([s], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=100)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ind, i in enumerate(ans_tokens):
            ans_tokens[ind] = ans_tokens[ind].replace('_', ' ')[7:].split('<')[0]

        return ' '.join(ans_tokens)




class SpellCheckModelNeuSpell(SpellCheckModelE2E):

    def __init__(self, max_count: Optional[int] = None):
        super().__init__(max_count)

    def __str__(self):
        return 'NeuSpell BERT'

    def correct_string(self, s) -> str:
        path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
        # path_prefix = '/home/ubuntu/omelnikov/'
        with open(path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.noise') as x:
            p = x.readlines()
        with open(path_prefix + 'grazie/spell/main/data/experiments/neuspell_bert/result_4.txt') as y:
            q = y.readlines()

        for i, j in zip(p, q):
            # print(i)
            # print(j)
            # print(s)
            # time.sleep(5)
            if s == i[:-1]:
                return j[:-1]

        print('WTF')
        print(s)
        time.sleep(5)
        return '--BUG--'





class SpellCheckModel(SpellCheckModelBase):
    def __init__(self, detector: BaseDetector, candidator: BaseCandidator, ranker: SpellRanker, max_count: Optional[int] = None):
        super().__init__(max_count)
        self.detector = detector
        self.candidator = candidator
        self.ranker = ranker

    def check(self, text: str, max_count: Optional[int] = None, round_digits: int = None, **kwargs) -> List[SpellCheckResult]:
        # что значит конструкция int or int?
        max_count = max_count or self.max_count
        round_digits = round_digits or 100

        spelled_words = self.detector.detect(text, **kwargs)

        # if not isinstance(self.candidator, AggregatedCandidator):

        all_candidates = self.candidator.get_candidates(text, spelled_words, **kwargs)

        scored_candidates = []
        for i, (spelled_word, candidates) in enumerate(zip(spelled_words, all_candidates)):
            scores = self.ranker.rank(text, spelled_word, candidates, **kwargs)
            variants = [SpellCheckVariant(candidate, round(score, round_digits), False) for score, candidate in
                        sorted(zip(scores, candidates), reverse=True)]

            spell_check_result = SpellCheckResult(spelled_word.interval[0], spelled_word.interval[1],
                                                  variants[:max_count])
            scored_candidates.append(spell_check_result)

        assert len(spelled_words) == len(scored_candidates)

        # else:
        #     scored_candidates = []
        #     for i, spelled_word in enumerate(spelled_words):
        #         canidator_ind = 0
        #
        #         candidates = []
        #         variants = []
        #         while canidator_ind < len(self.candidator._candidators):
        #
        #             cur_candidates = self.candidator.get_candidates_by_candidator(text, [spelled_word], canidator_ind, **kwargs)[0]
        #             candidates.extend(cur_candidates)
        #             scores = self.ranker.rank(text, spelled_word, candidates, **kwargs)
        #             variants = [SpellCheckVariant(candidate, round(score, round_digits), False) for score, candidate in
        #                         sorted(zip(scores, candidates), reverse=True)]
        #             if variants[0].score < 0.8:
        #                 print('Low score on sample', spelled_word.word)
        #                 canidator_ind += 1
        #             else:
        #                 break
        #
        #
        #
        #
        #         spell_check_result = SpellCheckResult(spelled_word.interval[0], spelled_word.interval[1],
        #                                               variants[:max_count])
        #         scored_candidates.append(spell_check_result)
        #
        #     assert len(spelled_words) == len(scored_candidates)

        return scored_candidates


# class SpellCheckModelT5(SpellCheckModelE2E):
#
#     def __init__(self, max_count: Optional[int] = None):
#         super().__init__(max_count)
#         self.beam_settings = TTSettings(num_beams=5, min_length=1, max_length=20)
#         self.model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
#
#     def __str__(self):
#         return 'T5'
#
#     def correct_string(self, s) -> str:
#         result = self.model.generate_text(f"grammar: {s}", args=self.beam_settings)
#         return result.text


if __name__ == '__main__':
    # path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
    path_prefix = '/home/ubuntu/omelnikov/'
    model = SpellCheckModelCharBasedTransformerMedium(checkpoint=path_prefix + 'grazie/spell/main/training/model_small_2_4.pt')
    # model = SpellCheckModelT5()

    # query = 'So I think we would not be live if our ancestors did not develop siences and tecnologies.'
    query = 'I WANT TO THAK YOU FOR PREPARING SUCH A GOOD PROGRAMME FOR US AND ESPECIALLY FOR TAKING US ON THE RIVER TRIP TO GREENWICH.'
    print('Query:', query)
    print('Result:', model.correct_string(query))
