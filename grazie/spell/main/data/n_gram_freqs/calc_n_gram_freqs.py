from typing import Tuple
from collections import Counter
from grazie.spell.main.preprocessing.tokenizer import SyntokTextTokenizer
import argparse


def calc_n_grams_freqs(texts_path: str, ngrams_path: str, n: int = 2, size: int = None) -> None:
    tokenizer = SyntokTextTokenizer()
    res: Counter[Tuple[str]] = Counter()
    with open(texts_path) as f:
        for line in f:
            text = line[:-1].lower()
            words = tokenizer.tokenize(text)
            for ind, word in enumerate(words):
                if ind <= len(words) - n:
                    n_gram = tuple(words[ind: ind + n])
                    if n_gram in res.keys():
                        res[n_gram] += 1
                    else:
                        res[n_gram] = 1

    with open(ngrams_path, 'w') as f:
        for key, value in res.most_common(size):
            f.write(' '.join(key) + ',' + str(value) + '\n')


def main():

    # Я не понял в чем удобство делать через argparse, в итоге ни по кнопке, ни через терминал не запускается
    # Из-за путей

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--texts_corpus", type=str, required=True)
    # parser.add_argument("--ngrams_path", type=str, required=True)
    # parser.add_argument("--n", type=int, required=True)
    #
    # args = parser.parse_args()
    # print(args.texts_corpus, args.ngrams_path, args.n)


    calc_n_grams_freqs('/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea60k', '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/2_grams.csv', n=2, size=10000)
    calc_n_grams_freqs('/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea60k', '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/n_gram_freqs/3_grams.csv', n=3, size=10000)


if __name__ == '__main__':
    main()
