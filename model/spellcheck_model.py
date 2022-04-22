import random
from abc import abstractmethod, ABC
from typing import List
from transformers import RobertaTokenizer
import torch
import time
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import transformers
from model.detector import HunspellDetector, BERTDetector
from data_utils.utils import get_texts_from_file

# PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


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


class CharBasedTransformerChecker(SpellCheckModelBase):

    class BartTokenizer(RobertaTokenizer):
        vocab_files_names = {"vocab_file": PATH_PREFIX + "data_utils/char_based_transformer_vocab/vocab.json",
                             "merges_file": PATH_PREFIX + "data_utils/char_based_transformer_vocab/merges.txt"}

    def __init__(self, config: dict = None, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = CharBasedTransformerChecker.BartTokenizer(
            PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_vocab.json",
            PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_merges.txt"
        )
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
        with open(PATH_PREFIX + 'dataset/bea/bea60k.noise') as x:
            p = x.readlines()
        with open(PATH_PREFIX + 'experiments/neuspell_bert/result_4.txt') as y:
            q = y.readlines()

        for i, j in zip(p, q):
            if text == i[:-1]:
                return j[:-1]

        print(f'Text "{text}" not found in NeuSpell result file')
        time.sleep(5)
        return '--BUG--'


class BartChecker(SpellCheckModelBase):

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
                    self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.model = self.model.to(self.device)
        print(f'Device: {self.device}')

    def from_pretrained(self):
        self.model = BartForConditionalGeneration.from_pretrained("melnikoff-oleg/bart-end-to-end")

    def __str__(self):
        return f'BART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:
        ans_ids = self.model.generate(self.tokenizer([text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return ' '.join(ans_tokens)


class BertBartChecker(SpellCheckModelBase):

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
        return f'SepMaskBART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:
        init_text = text
        if text.upper() == text:
            text = text.lower()
        detector = HunspellDetector()
        # detector = BERTDetector()
        try:
            spells = detector.detect(text)
        except Exception as e:
            print(e)
            return text
        col = 0

        while len(spells) > 0 and col < 1:
            spell_ind = 0
            # spell_ind = random.randint(0, len(spells) - 1)
            new_text = spells[spell_ind].word + ' <sep> ' + text[: spells[spell_ind].interval[0]] + '<mask>'+ text[spells[spell_ind].interval[1]: ]
            if col > 0:
                print('Init tex:', init_text)
                print('Cur text:', text)
                print(f'New spell:|{spells[spell_ind].word}|')
            ans_ids = self.model.generate(self.tokenizer([new_text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
            ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text = ' '.join(ans_tokens)
            spells = detector.detect(text)
            col += 1

        if col > 1:
            print('Init text:', init_text)
            print('Cur text:', text)
            print()

        if init_text == init_text.upper():
            text = text.upper()
        return text


class CharBasedSepMask(SpellCheckModelBase):

    class BartTokenizer(RobertaTokenizer):
        vocab_files_names = {"vocab_file": PATH_PREFIX + "data_utils/char_based_transformer_vocab/vocab.json",
                             "merges_file": PATH_PREFIX + "data_utils/char_based_transformer_vocab/merges.txt"}

    def __init__(self, config: dict = None, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = CharBasedTransformerChecker.BartTokenizer(
            PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_vocab.json",
            PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_merges.txt"
        )
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
        init_text = text
        if text.upper() == text:
            text = text.lower()
        detector = HunspellDetector()
        # detector = BERTDetector()
        try:
            spells = detector.detect(text)
        except Exception as e:
            print(e)
            return text


        col = 0

        while len(spells) > 0 and col < 1:
            spell_ind = 0
            # spell_ind = random.randint(0, len(spells) - 1)
            new_text = spells[spell_ind].word + ' <sep> ' + text[: spells[spell_ind].interval[0]] + '<mask>'+ text[spells[spell_ind].interval[1]: ]
            if col > 0:
                print('Init tex:', init_text)
                print('Cur text:', text)
                print(f'New spell:|{spells[spell_ind].word}|')
            new_text = new_text.replace(' ', '_')
            ans_ids = self.model.generate(self.tokenizer([new_text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
            ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for ind, i in enumerate(ans_tokens):
                ans_tokens[ind] = ans_tokens[ind].replace('_', ' ')[7:].split('<')[0]
            text = ' '.join(ans_tokens)
            spells = detector.detect(text)
            col += 1

        if col > 1:
            print('Init text:', init_text)
            print('Cur text:', text)
            print()

        if init_text == init_text.upper():
            text = text.upper()
        return text


class MaskWordBartChecker(SpellCheckModelBase):

    def __init__(self, checkpoint: str = 'No learning', model: BartForConditionalGeneration = None,
                 device: torch.device = None):
        self.checkpoint = checkpoint
        transformers.set_seed(42)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print('MaskWordBART device:', self.device)

        if model is not None:
            self.model = model
        else:
            if checkpoint == 'No learning':
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                config = BartForConditionalGeneration.from_pretrained('facebook/bart-base').config
                self.model = BartForConditionalGeneration(config)
                self.model = self.model.to(self.device)
                # Model was trained on GPU, maybe we are inferring on CPU
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                else:
                    self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.model = self.model.to(self.device)

    def __str__(self):
        return f'BART, checkpoint: {self.checkpoint.split("/")[-1]}'

    def correct(self, text: str) -> str:

        detector = HunspellDetector()
        # detector = BERTDetector()

        spells = detector.detect(text)
        col = 0
        while len(spells) > 0 and col < 1:
            # spell_ind = 0
            spell_ind = random.randint(0, len(spells) - 1)
            new_text = spells[spell_ind].word + ' <sep> ' + text[: spells[spell_ind].interval[0]] + '<mask>'+ text[spells[spell_ind].interval[1]: ]
            print(f'Task: |{new_text}|')
            ans_ids = self.model.generate(self.tokenizer([new_text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=500)
            ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            corr_word = ' '.join(ans_tokens)
            print( f'Corr word: |{corr_word}|')
            text = text[: spells[spell_ind].interval[0]] + corr_word + text[spells[spell_ind].interval[1]: ]
            spells = HunspellDetector().detect(text)
            col += 1

        return text


def spellcheck_model_test(model: SpellCheckModelBase):
    text_noise = 'Moroever I have chosen this month because I think the weather will be fine.'
    text_gt = 'Moreover I have chosen this month because I think the weather will be fine.'
    text_result = model.correct(text_noise)
    print(f'\nSpellcheck model testing\n\nModel: {str(model)}\n\n{text_noise} - Noised text\n{text_gt} - GT text'
          f'\n{text_result} - Result text')


def char_based_model_check():
    # checkpoint = PATH_PREFIX + 'training/checkpoints/model_big_0_9.pt'
    checkpoint = 'No learning'
    d_model = 256
    model = CharBasedTransformerChecker(config={'d_model': d_model, 'encoder_layers': 6, 'decoder_layers': 6,
                                         'encoder_attention_heads': 8, 'decoder_attention_heads': 8,
                                         'encoder_ffn_dim': d_model * 4, 'decoder_ffn_dim': d_model * 4},
                                 checkpoint=checkpoint)
    spellcheck_model_test(model)


def bart_pretrain_test():
    # model = BartChecker(model=BartForConditionalGeneration.from_pretrained('facebook/bart-base'))
    checker = BartChecker()
    checker.from_pretrained()

    """ spell correction """
    print(checker.correct("I luk foward to receving your reply"))
    # → "I look forward to receiving your reply"
    print(checker.correct_strings(["I luk foward to receving your reply", ]))
    # → ["I look forward to receiving your reply"]

    """ evaluation of models """
    # texts_gt, texts_noise = get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea2.gt'), \
    #                         get_texts_from_file(PATH_PREFIX + 'dataset/bea/bea2.noise')
    # evaluate(checker, texts_gt=texts_gt, texts_noise=texts_noise)
    # spellcheck_model_test(checker)


def bart_sep_mask_test():

    model = BertBartChecker(model=...)
    # model.from_pretrained()


    # model_name = 'bart-sep-mask_v1_3'
    # checkpoint = f'training/checkpoints/{model_name}'
    # model = SepMaskBART(checkpoint=PATH_PREFIX + checkpoint + '.pt', device=torch.device('cuda:1'))

    spellcheck_model_test(model)


def bart_mask_word_test():
    # model = MaskWordBART(model=BartForConditionalGeneration.from_pretrained('facebook/bart-base'))
    for i in range(4, 8):
        try:
            model = MaskWordBartChecker(checkpoint=PATH_PREFIX + 'training/checkpoints/bart-mask-word_v1_3.pt',
                                 device=torch.device(f'cuda:{i}'))
            spellcheck_model_test(model)
            return
        except Exception as e:
            print(e)
            raise e
            # continue
    print('All GPUs bad')


if __name__ == '__main__':
    bart_pretrain_test()
