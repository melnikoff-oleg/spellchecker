from tqdm import tqdm

if __name__ == '__main__':
    PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'
    # PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/'

    dataset_paths = ['data/datasets/1blm/1blm.train', 'data/datasets/1blm/1blm.test']

    for dataset_path in dataset_paths:
        with open(PATH_PREFIX + dataset_path + '.noise') as f:
            with open(PATH_PREFIX + dataset_path + '.gt') as g:
                with open(PATH_PREFIX + dataset_path + '.noise.bart_sep_mask', 'w') as t:
                    for i, j in tqdm(zip(f.readlines(), g.readlines())):
                        words_noise, words_gt = i[:-1].split(' '), j[:-1].split(' ')
                        for ind in range(len(words_noise)):
                            if words_noise[ind] != words_gt[ind] or ind == len(words_noise) - 1:
                                noisy_word = words_noise[ind]
                                words_noise[ind] = '<mask>'
                                task = noisy_word + ' <sep> ' + ' '.join(words_noise)
                                t.write(task + '\n')
                                break
