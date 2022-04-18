# PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
PATH_PREFIX = '//'


def create_short_version_of_dataset(length: int, noise_old: str, gt_old: str, noise_new: str, gt_new: str):
    dataset_pairs = [(noise_old, noise_new), (gt_old, gt_new)]
    for old, new in dataset_pairs:
        with open(PATH_PREFIX + old) as f:
            with open(PATH_PREFIX + new, 'w') as g:
                ind = 0
                for line in f:
                    g.write(line)
                    ind += 1
                    if ind == length:
                        break


def test():
    create_short_version_of_dataset(2, 'datasets/bea/bea60k.noise', 'datasets/bea/bea60k.gt', 'datasets/bea/bea2.noise',
                                    'datasets/bea/bea2.gt')


if __name__ == '__main__':
    test()
