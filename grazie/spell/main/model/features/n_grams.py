from typing import List

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature
from typing import Dict, Tuple
from time import sleep
from grazie.spell.main.preprocessing.tokenizer import SyntokTextTokenizer


class NGramsFeature(BaseFeature):
    def __init__(self, n=1):
        self.n = n

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        tokenizer = SyntokTextTokenizer()
        ngram_freqs: Dict[Tuple[str], int] = {}
        with open(f'/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/{self.n}_grams.csv') as f:
            for line in f:
                text_line = line[:-1]
                words = ','.join(text_line.split(',')[:-1])
                tokens = words.split(' ')
                # print(words, tokens)
                # print(self.n)
                # sleep(1)
                freq = int(text_line.split(',')[-1])
                ngram_freqs[tuple(tokens)] = freq

        tokens = tokenizer.tokenize(text)
        try:
            ind = tokens.index(spelled_word.word)
        except Exception as e:
            print(tokens, '\n', spelled_word.word)
            return [0 for candidate in candidates]
            # raise e
        res = []
        for candidate in candidates:
            tokens[ind] = candidate
            avg = 0
            cnt = 0
            for st in range(ind - self.n + 1, ind + 1):
                if st >= 0 and st + self.n <= len(tokens):
                    cnt += 1
                    key = tuple(tokens[st: st + self.n])
                    if key in ngram_freqs.keys():
                        avg += ngram_freqs[key]
                    else:
                        avg += 0
            if cnt > 0:
                res.append(avg / cnt)
            else:
                res.append(0)
        return res


class BiGramsFeature(NGramsFeature):
    def __init__(self):
        super(NGramsFeature, self).__init__()
        self.n = 2


class TriGramsFeature(NGramsFeature):
    def __init__(self):
        super(NGramsFeature, self).__init__()
        self.n = 3
