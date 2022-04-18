from tqdm import tqdm
import random
# PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellcheker/'


def shuffle_dataset(noise_old: str, gt_old: str, noise_new: str, gt_new: str):
    with open(PATH_PREFIX + noise_old) as f:
        with open(PATH_PREFIX + gt_old) as g:
            with open(PATH_PREFIX + noise_new, 'w') as t:
                with open(PATH_PREFIX + gt_new, 'w') as s:
                    sent_pairs = list(zip(f.readlines(), g.readlines()))
                    random.shuffle(sent_pairs)
                    for i, j in tqdm(sent_pairs):
                        t.write(i)
                        s.write(j)


def test():
    shuffle_dataset('datasets/bea/bea10.noise', 'datasets/bea/bea10.gt', 'datasets/bea/bea10.noise.shuffled',
                    'datasets/bea/bea10.gt.shuffled')


if __name__ == '__main__':
    test()
