PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'


def check_tokenization(noise: str, gt: str):
    with open(PATH_PREFIX + noise) as f:
        noise_texts = f.readlines()
    with open(PATH_PREFIX + gt) as f:
        gt_texts = f.readlines()
    ttl_sents = 0
    broken_tokenization_sents = 0
    for i, j in zip(noise_texts, gt_texts):
        ttl_sents += 1
        if len(i.split(' ')) != len(j.split(' ')):
            broken_tokenization_sents += 1
    print(f'Total sentences: {ttl_sents}, Broken tokenization: {broken_tokenization_sents}')
    print(f'Part with broken tokenization: {round(broken_tokenization_sents / ttl_sents, 2)}')


def test():
    check_tokenization(noise='datasets/1blm/1blm.train.noise', gt='datasets/1blm/1blm.train.gt')


if __name__ == '__main__':
    test()
