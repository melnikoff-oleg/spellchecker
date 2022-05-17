from tqdm import tqdm
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'


def create_dataset_for_oldbart_finetune(noise_old: str, gt_old: str, noise_new: str, gt_new: str):

    with open(PATH_PREFIX + noise_old) as f:
        with open(PATH_PREFIX + gt_old) as g:
            with open(PATH_PREFIX + noise_new, 'w') as t:
                with open(PATH_PREFIX + gt_new, 'w') as s:
                    print_one_sample = True
                    for i, j in tqdm(zip(f.readlines(), g.readlines())):
                        words_noise, words_gt = i[:-1].split(' '), j[:-1].split(' ')
                        if len(words_noise) == len(words_gt):
                            was = False
                            for ind in range(len(words_noise)):
                                if words_noise[ind] != words_gt[ind] or (ind == len(words_noise) - 1 and not was):
                                    was = True
                                    words_noise[ind] = '<mask>'
                            text_noise = ' '.join(words_noise) + '\n'
                            text_gt = ' '.join(words_gt) + '\n'
                            if print_one_sample:
                                print(f'Text noise - |{text_noise}|')
                                print(f'Text gt - |{text_gt}|')
                                print()
                                print_one_sample = False
                            t.write(text_noise)
                            s.write(text_gt)


def main():
    create_dataset_for_oldbart_finetune('dataset/1blm/1blm.train.noise', 'dataset/1blm/1blm.train.gt',
                                'dataset/1blm/1blm.train.noise.oldbart',
                                'dataset/1blm/1blm.train.gt.oldbart')
    create_dataset_for_oldbart_finetune('dataset/1blm/1blm.test.noise', 'dataset/1blm/1blm.test.gt',
                                'dataset/1blm/1blm.test.noise.oldbart',
                                'dataset/1blm/1blm.test.gt.oldbart')


if __name__ == '__main__':
    main()
