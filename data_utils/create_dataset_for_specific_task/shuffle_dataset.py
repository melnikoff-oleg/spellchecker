from tqdm import tqdm
import random

if __name__ == '__main__':
    PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
    # PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/'

    dataset_paths = ['data/datasets/1blm/1blm.train', 'data/datasets/1blm/1blm.test']
    # dataset_paths = ['data/datasets/bea/bea10', 'data/datasets/bea/bea10']

    for dataset_path in dataset_paths:
        with open(PATH_PREFIX + dataset_path + '.noise.sep_mask') as f:
            with open(PATH_PREFIX + dataset_path + '.gt.sep_mask') as g:
                with open(PATH_PREFIX + dataset_path + '.noise.sep_mask.shuffled', 'w') as t:
                    with open(PATH_PREFIX + dataset_path + '.gt.sep_mask.shuffled', 'w') as s:
                        ttl = 0
                        sent_pairs = list(zip(f.readlines(), g.readlines()))
                        random.shuffle(sent_pairs)
                        for i, j in tqdm(sent_pairs):
                            ttl += 1
                            # if ttl > 10:
                            #     break
                            t.write(i)
                            s.write(j)
                            # print(i)
                            # print(j)
                            # print()
                        print('ttl:', ttl)
