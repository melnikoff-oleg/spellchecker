from typing import List

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature
from typing import Dict, Tuple
from grazie.spell.main.preprocessing.tokenizer import SyntokTextTokenizer
from grazie.spell.test.feature_tester import test_feature


class NGramsFeature(BaseFeature):
    def __init__(self, ngrams_path, n: int = 1):
        self.ngrams_path = ngrams_path
        self.n = n
        self.tokenizer = SyntokTextTokenizer()
        self.ngram_freqs: Dict[Tuple[str], int] = {}

        self.load_ngrams()

    def load_ngrams(self):
        with open(self.ngrams_path) as f:
            for line in f:
                text_line = line[:-1]
                words = ','.join(text_line.split(',')[:-1])
                tokens = words.split(' ')
                freq = int(text_line.split(',')[-1])
                self.ngram_freqs[tuple(tokens)] = freq

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        tokens = self.tokenizer.tokenize(text)

        if spelled_word.word in tokens:
            ind = tokens.index(spelled_word.word)
        else:
            print(tokens, '\n', spelled_word.word)
            return [0 for candidate in candidates]

        # lower casing all token for normal n-grams behaviour
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        res = []
        for candidate in candidates:
            tokens[ind] = candidate.lower()
            avg = 0
            cnt = 0
            for st in range(ind - self.n + 1, ind + 1):
                if st >= 0 and st + self.n <= len(tokens):
                    cnt += 1
                    key = tuple(tokens[st: st + self.n])
                    if key in self.ngram_freqs.keys():
                        avg += self.ngram_freqs[key]
                    else:
                        avg += 0
            if cnt > 0:
                res.append(avg / cnt)
            else:
                res.append(0)
        return res


def run_tests():
    for n in range(1, 4):
        print(f'Testing {n}-grams')
        test_feature(NGramsFeature(ngrams_path=f'/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/{n}_grams.csv', n=n))
        print('\n')


if __name__ == '__main__':
    run_tests()
