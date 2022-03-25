from abc import abstractmethod, ABC
from typing import List
from transformers import RobertaTokenizer
import string
import torch
import json
import time
from transformers import BartConfig, BartForConditionalGeneration


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

    def __init__(self, config: dict, checkpoint: str):
        self.config = config
        self.checkpoint = checkpoint
        CharBasedTransformer.create_vocab_files()
        tokenizer = CharBasedTransformer.BartTokenizer("url_vocab.json", "url_merges.txt")
        config['vocab_size'] = tokenizer.vocab_size
        model_config = BartConfig(**config)
        model = BartForConditionalGeneration(model_config)

        # Model was trained on GPU, maybe we are inferring on CPU
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
        return f'Char-Based Transformer, checkpoint: {self.checkpoint.split("/")[0]}'

    def correct(self, text: str) -> str:
        text = text.replace(' ', '_')
        ans_ids = self.model.generate(self.tokenizer([text], return_tensors='pt').to(self.device)["input_ids"],
                                      num_beams=5, min_length=5, max_length=100)
        ans_tokens = self.tokenizer.batch_decode(ans_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ind, i in enumerate(ans_tokens):
            ans_tokens[ind] = ans_tokens[ind].replace('_', ' ')[7:].split('<')[0]
        return ' '.join(ans_tokens)


class SpellCheckModelNeuSpell(SpellCheckModelBase):

    def __str__(self):
        return 'NeuSpell BERT'

    def correct(self, text: str) -> str:
        path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
        # path_prefix = '/home/ubuntu/omelnikov/'
        with open(path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.noise') as x:
            p = x.readlines()
        with open(path_prefix + 'grazie/spell/main/data/experiments/neuspell_bert/result_4.txt') as y:
            q = y.readlines()

        for i, j in zip(p, q):
            if text == i[:-1]:
                return j[:-1]

        print(f'Text "{text}" not found in NeuSpell result file')
        time.sleep(5)
        return '--BUG--'


def spellcheck_model_test():
    path_prefix = '/home/ubuntu/omelnikov/grazie/spell/main/'
    model = CharBasedTransformer(config={'d_model': 256, 'encoder_layers': 6, 'decoder_layers': 6,
                                         'encoder_attention_heads': 8, 'decoder_attention_heads': 8,
                                         'encoder_ffn_dim': 1024, 'decoder_ffn_dim': 1024},
                                 checkpoint=path_prefix + 'training/model_big_0_9.pt')

    text_noise = 'I was trully dissapointed by it.'
    text_gt = 'I was truly disappointed by it.'
    text_result = model.correct(text_noise)
    print(f'\nSpellcheck model testing\n\nModel: {str(model)}\n\n{text_noise} - Noised text\n{text_gt} - GT text\n{text_result} - Result text')


if __name__ == '__main__':
    spellcheck_model_test()
