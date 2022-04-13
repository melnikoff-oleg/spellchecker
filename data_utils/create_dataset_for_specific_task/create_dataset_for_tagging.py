from tqdm import tqdm


# Example
# {'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
#  'id': '0',
#  'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0],
#  'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
#  'tokens': ['EU',
#   'rejects',
#   'German',
#   'call',
#   'to',
#   'boycott',
#   'British',
#   'lamb',
#   '.']}

# My case
# id: ...
# labels: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
# tokens: [...]
#
# in file like this: {"a": 1, "b": 2.0, "c": "foo", "d": false}
# {"a": 4, "b": -5.5, "c": null, "d": true}
# object by object line by line
#

import json
import nltk

def main():
    # PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/'
    PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
    datasets_paths = ['1blm/1blm.train', '1blm/1blm.test', 'bea/bea500']
    for dataset_path in datasets_paths:
        with open(PATH_PREFIX + f'data/datasets/{dataset_path}.noise') as f:
            with open(PATH_PREFIX + f'data/datasets/{dataset_path}.gt') as g:
                with open(PATH_PREFIX + f'data/datasets/{dataset_path}.tagging', 'w') as r:
                    a = f.readlines()
                    b = g.readlines()
                    ttl = 0
                    for i, j in tqdm(zip(a, b), total=len(a)):
                        x, y = i[:-1], j[:-1]
                        p, q = nltk.word_tokenize(x), nltk.word_tokenize(y)
                        if len(p) == len(q):
                            labels = []
                            for t, k in zip(p, q):
                                if t != k:
                                    labels.append(1)
                                else:
                                    labels.append(0)

                            js = {"id": ttl, "labels": labels, "tokens": p, "sent": x}
                            r.write(json.dumps(js) + "\n")

                            ttl += 1

                    print(f'In new dataset: {ttl}, which is {round(ttl / len(a), 2)}')


if __name__ == '__main__':
    nltk.download('punkt')
    main()
