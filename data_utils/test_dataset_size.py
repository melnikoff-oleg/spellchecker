PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'


def check_size(path: str):
    with open(PATH_PREFIX + path) as f:
        texts = f.readlines()
    print(f'Total sentences: {len(texts)}')


def test():
    check_size(path='dataset/1blm/1blm.train.noise.sep_mask_all_sent')


if __name__ == '__main__':
    test()
