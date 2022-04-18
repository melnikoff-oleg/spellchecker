from tqdm import tqdm
import random
# PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellcheker/'


def create_dataset_for_sep_mask(noise_old: str, gt_old: str, noise_new: str, gt_new: str):

    with open(PATH_PREFIX + noise_old) as f:
        with open(PATH_PREFIX + gt_old) as g:
            with open(PATH_PREFIX + noise_new, 'w') as t:
                with open(PATH_PREFIX + gt_new, 'w') as s:
                    init_sents = 0
                    new_sents = 0
                    sent_pairs = []
                    for i, j in tqdm(zip(f.readlines(), g.readlines())):
                        init_sents += 1
                        words_noise, words_gt = i[:-1].split(' '), j[:-1].split(' ')
                        if len(words_noise) == len(words_gt):
                            was = False
                            for ind in range(len(words_noise)):
                                if words_noise[ind] != words_gt[ind] or (ind == len(words_noise) - 1 and not was):
                                    was = True
                                    new_sents += 1
                                    noisy_word = words_noise[ind]
                                    words_noise[ind] = '<mask>'
                                    task = noisy_word + ' <sep> ' + ' '.join(words_noise)
                                    words_noise[ind] = words_gt[ind]
                                    gt = ' '.join(words_noise)
                                    sent_pairs.append([task + '\n', gt + '\n'])
                                    words_noise[ind] = noisy_word
                    random.shuffle(sent_pairs)
                    for i, j in tqdm(sent_pairs):
                        t.write(i)
                        s.write(j)
                    print(f'Init number of sentences: {init_sents}\nNew number of sentences: {new_sents}')


def test():
    create_dataset_for_sep_mask('datasets/bea/bea60k.noise', 'datasets/bea/bea60k.gt',
                                'datasets/bea/bea60k.noise.sep_mask', 'datasets/bea/bea60k.gt.sep_mask')


if __name__ == '__main__':
    test()
