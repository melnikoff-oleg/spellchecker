from typing import Tuple
from collections import Counter
from grazie.spell.main.preprocessing.tokenizer import SyntokTextTokenizer


def calc_n_grams_freqs(texts_path: str, n: int = 2, size: int = None) -> None:
    tokenizer = SyntokTextTokenizer()
    res: Counter[Tuple[str]] = Counter()
    with open(texts_path) as f:
        for line in f:
            text = line[:-1].lower()
            # words = text.split(' ')
            words = tokenizer.tokenize(text)
            for ind, word in enumerate(words):
                if ind <= len(words) - n:
                    n_gram = tuple(words[ind: ind + n])
                    if n_gram in res.keys():
                        res[n_gram] += 1
                    else:
                        res[n_gram] = 1
    #                     /Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/
    with open(str(n) + '_grams.csv', 'w') as f:
        for key, value in res.most_common(size):
            f.write(' '.join(key) + ',' + str(value) + '\n')


def main():
    calc_n_grams_freqs('/grazie/spell/main/data/datasets/test.bea60k', n=2, size=10000)
    calc_n_grams_freqs('/grazie/spell/main/data/datasets/test.bea60k', n=3, size=10000)


if __name__ == '__main__':
    main()
