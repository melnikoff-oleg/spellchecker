from abc import abstractmethod, ABC
from typing import List
from transformers import RobertaTokenizer
import string
import torch
import json
import time
from tqdm import tqdm
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import transformers
from grazie.spell.main.model.detector import BaseDetector
from grazie.spell.main.model.ranker import SpellRanker
from grazie.common.main.ranking.catboost_ranker import CatBoostRanker
from grazie.spell.main.model.spellcheck_model_old import SpellCheckVariant


from grazie.common.main.ranking.ranker import RankQuery, RankVariant
from grazie.spell.main.data.base import SpelledText
from grazie.spell.main.data.utils import get_test_data
from grazie.spell.main.model.candidator import BaseCandidator, AggregatedCandidator, IdealCandidator, HunspellCandidator
from grazie.spell.main.model.detector import IdealDetector, HunspellDetector
from grazie.spell.main.model.features.features_collector import FeaturesCollector
from grazie.spell.main.model.ranker import FeaturesSpellRanker


# PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/'

class SpellCheckModelBase(ABC):

    @abstractmethod
    def correct(self, text: str) -> str:
        raise NotImplementedError

    def correct_strings(self, texts: List[str]) -> List[str]:
        return [self.correct(text) for text in texts]

    def correct_from_file(self, src: str, dest: str):
        with open(src) as src_texts:
            with open(dest) as dest_texts:
                for text in src_texts:
                    dest_texts.write(self.correct(text[:-1]) + '\n')


class CharBasedTransformer(SpellCheckModelBase):

    @classmethod
    def create_vocab_files(cls):
        chars = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + list(string.punctuation) + list(string.digits) + \
                list(string.ascii_lowercase) + list(string.ascii_uppercase)
        url_vocab = {c: i for i, c in enumerate(chars)}
        with open("url_vocab.json", 'w') as json_file:
            json.dump(url_vocab, json_file)
        merges = "#version: 0.2\n"
        with open("url_merges.txt", 'w') as f:
            f.write(merges)

    class BartTokenizer(RobertaTokenizer):
        vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

    def __init__(self, config: dict = None, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        CharBasedTransformer.create_vocab_files()
        self.tokenizer = CharBasedTransformer.BartTokenizer("url_vocab.json", "url_merges.txt")
        if not config is None:
            config['vocab_size'] = self.tokenizer.vocab_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is not None:
            self.model = model
        else:
            model_config = BartConfig(**config)
            self.model = BartForConditionalGeneration(model_config)
            if checkpoint != 'No learning':
                # Model was trained on GPU, maybe we are inferring on CPU
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint))

        self.model = self.model.to(self.device)

    def __str__(self):
        return f'Char-Based Transformer, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:
        text = text.replace(' ', '_')
        ans_ids = self.model.generate(self.tokenizer([text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ind, i in enumerate(ans_tokens):
            ans_tokens[ind] = ans_tokens[ind].replace('_', ' ')[7:].split('<')[0]
        return ' '.join(ans_tokens)


class SpellCheckModelNeuSpell(SpellCheckModelBase):

    def __str__(self):
        return 'NeuSpell BERT'

    def correct(self, text: str) -> str:
        with open(PATH_PREFIX + 'data/datasets/bea/bea60k.noise') as x:
            p = x.readlines()
        with open(PATH_PREFIX + 'data/experiments/neuspell_bert/result_4.txt') as y:
            q = y.readlines()

        for i, j in zip(p, q):
            if text == i[:-1]:
                return j[:-1]

        print(f'Text "{text}" not found in NeuSpell result file')
        time.sleep(5)
        return '--BUG--'


class BART(SpellCheckModelBase):

    def __init__(self, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is not None:
            self.model = model
        else:
            if checkpoint == 'No learning':
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
                self.model = BartForConditionalGeneration(config)
                # Model was trained on GPU, maybe we are inferring on CPU
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint))

        self.model = self.model.to(self.device)

    def __str__(self):
        return f'BART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:
        ans_ids = self.model.generate(self.tokenizer([text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return ' '.join(ans_tokens)


class SepMaskBART(SpellCheckModelBase):

    def __init__(self, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is not None:
            self.model = model
        else:
            if checkpoint == 'No learning':
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
                self.model = BartForConditionalGeneration(config)
                # Model was trained on GPU, maybe we are inferring on CPU
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint))

        self.model = self.model.to(self.device)

    def __str__(self):
        return f'BART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:

        spells = HunspellDetector().detect(text)
        col = 0
        while len(spells) > 0 and col < 1:
            new_text = spells[0].word + ' <sep> ' + text[: spells[0].interval[0]] + '<mask>'+ text[spells[0].interval[1]: ]

            ans_ids = self.model.generate(self.tokenizer([new_text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
            ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text = ' '.join(ans_tokens)
            spells = HunspellDetector().detect(text)
            col += 1

        return text


class MaskWordBART(SpellCheckModelBase):

    def __init__(self, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is not None:
            self.model = model
        else:
            if checkpoint == 'No learning':
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
                self.model = BartForConditionalGeneration(config)
                # Model was trained on GPU, maybe we are inferring on CPU
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint))

        self.model = self.model.to(self.device)

    def __str__(self):
        return f'BART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:

        spells = HunspellDetector().detect(text)
        col = 0
        while len(spells) > 0 and col < 1:
            new_text = spells[0].word + ' <sep> ' + text[: spells[0].interval[0]] + '<mask>'+ text[spells[0].interval[1]: ]

            ans_ids = self.model.generate(self.tokenizer([new_text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
            ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            corr_word = ' '.join(ans_tokens)
            text = text[: spells[0].interval[0]] + corr_word[:-1] + text[spells[0].interval[1]: ]
            spells = HunspellDetector().detect(text)
            col += 1

        return text


class DetectorCandidatorRankerModel(SpellCheckModelBase):

    def __init__(self, detector: BaseDetector, candidator: BaseCandidator, ranker: SpellRanker):
        self.detector = detector
        self.candidator = candidator
        self.ranker = ranker

    def correct(self, text: str) -> str:
        spelled_words = self.detector.detect(text)
        all_candidates = self.candidator.get_candidates(text, spelled_words)
        shift = 0
        for i, (spelled_word, candidates) in enumerate(zip(spelled_words, all_candidates)):
            scores = self.ranker.rank(text, spelled_word, candidates)
            variants = [SpellCheckVariant(candidate, score, False) for score, candidate in
                        sorted(zip(scores, candidates), reverse=True)]
            res_word = variants[0].substitution
            text = text[: spelled_word.interval[0] + shift] + res_word + text[spelled_word.interval[1] + shift: ]
            shift += len(res_word) - len(spelled_word.word)

        return text


def spellcheck_model_test(model: SpellCheckModelBase):
    # text_noise = 'I have just received your letter which made me so hapy. I can not belive that I won first prize in your competition, because I have always believed I am an unluky man and now I think some things are changing in my life.'
    # text_gt = 'I have just received your letter which made me so happy. I can not believe that I won first prize in your competition, because I have always believed I am an unlucky man and now I think some things are changing in my life.'
    text_noise = 'Moroever, I have choooosen this month because I think the weather will be fine.'
    text_gt = 'Moreover, I have chosen this month because I think the weather will be fine.'
    # text_noise = 'groat <sep> My friends are <mask> but they eat too many carbs.'
    # text_gt = 'My friends are great but they eat too many carbs.'
    text_result = model.correct(text_noise)
    print(f'\nSpellcheck model testing\n\nModel: {str(model)}\n\n{text_noise} - Noised text\n{text_gt} - GT text'
          f'\n{text_result} - Result text')


def char_based_model_check():
    checkpoint = 'training/model_big_0_9.pt'
    d_model = 256
    model = CharBasedTransformer(config={'d_model': d_model, 'encoder_layers': 6, 'decoder_layers': 6,
                                         'encoder_attention_heads': 8, 'decoder_attention_heads': 8,
                                         'encoder_ffn_dim': d_model * 4, 'decoder_ffn_dim': d_model * 4},
                                 checkpoint=PATH_PREFIX + checkpoint)
    spellcheck_model_test(model)


def bart_pretrain_test():
    model = BART(model=BartForConditionalGeneration.from_pretrained('facebook/bart-base'))
    spellcheck_model_test(model)


def bart_sep_mask_test():
    model = SepMaskBART(model=BartForConditionalGeneration.from_pretrained('facebook/bart-base'))
    spellcheck_model_test(model)


def bart_mask_word_test():
    model = MaskWordBART(model=BartForConditionalGeneration.from_pretrained('facebook/bart-base'))
    spellcheck_model_test(model)


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


def three_part_model_test():

    detector = HunspellDetector()
    candidator = HunspellCandidator()
    ranker = CatBoostRanker(iterations=100)

    freqs_table_path = PATH_PREFIX + 'data/n_gram_freqs/1_grams.csv'
    bigrams_table_path = PATH_PREFIX + 'data/n_gram_freqs/2_grams.csv'
    trigrams_table_path = PATH_PREFIX + 'data/n_gram_freqs/3_grams.csv'

    features_collector = FeaturesCollector(['bart_prob', 'levenshtein'], bigrams_table_path, trigrams_table_path,
                                           FeaturesCollector.load_freqs(freqs_table_path))
    train_gt = PATH_PREFIX + 'data/datasets/1blm/1blm.train.gt'
    train_noise = PATH_PREFIX + 'data/datasets/1blm/1blm.train.noise'
    train_data, test_data = get_test_data(train_gt, train_noise, size=20, train_part=0.5)
    train_rank_data = prepare_ranking_training_data(train_data, candidator, features_collector)
    test_rank_data = prepare_ranking_training_data(test_data, candidator, features_collector)

    ranker.fit(train_rank_data, test_rank_data, epochs=20, lr=3e-4, l2=0., l1=0.)
    model = DetectorCandidatorRankerModel(detector, candidator, FeaturesSpellRanker(features_collector, ranker))

    spellcheck_model_test(model)


if __name__ == '__main__':
    bart_mask_word_test()
