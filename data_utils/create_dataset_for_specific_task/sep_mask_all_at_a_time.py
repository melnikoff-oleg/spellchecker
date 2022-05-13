from tqdm import tqdm
import random
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'


def create_dataset_for_sep_mask(noise_old: str, gt_old: str, noise_new: str, gt_new: str, test_mode: bool = False,
                                sep_token: str = '</s>', sent_token: str = '</s>'):

    with open(PATH_PREFIX + noise_old) as f:
        with open(PATH_PREFIX + gt_old) as g:
            with open(PATH_PREFIX + noise_new, 'w') as t:
                with open(PATH_PREFIX + gt_new, 'w') as s:
                    ttl = 0
                    for i, j in tqdm(zip(f.readlines(), g.readlines())):
                        if test_mode and ttl == 10:
                            break
                        ttl += 1
                        words_noise, words_gt = i[:-1].split(' '), j[:-1].split(' ')
                        if len(words_noise) == len(words_gt):
                            was = False
                            words_before_sep = []
                            sent_with_masks = words_noise
                            for ind in range(len(words_noise)):
                                if words_noise[ind] != words_gt[ind] or (ind == len(words_noise) - 1 and not was):
                                    was = True
                                    words_before_sep.append(words_noise[ind])
                                    sent_with_masks[ind] = '<mask>'
                            text_noise = f' {sep_token} '.join(words_before_sep) + f' {sent_token} ' + \
                                         ' '.join(sent_with_masks) + '\n'
                            text_gt = ' '.join(words_gt) + '\n'
                            if test_mode:
                                print(f'Text noise - |{text_noise}|')
                                print(f'Text gt - |{text_gt}|')
                                print()
                            else:
                                t.write(text_noise)
                                s.write(text_gt)


def test():
    create_dataset_for_sep_mask('dataset/1blm/1blm.train.noise', 'dataset/1blm/1blm.train.gt',
                                'dataset/1blm/1blm.train.noise.sep_mask_all_sent',
                                'dataset/1blm/1blm.train.gt.sep_mask_all_sent')
    create_dataset_for_sep_mask('dataset/1blm/1blm.test.noise', 'dataset/1blm/1blm.test.gt',
                                'dataset/1blm/1blm.test.noise.sep_mask_all_sent',
                                'dataset/1blm/1blm.test.gt.sep_mask_all_sent')

    # create_dataset_for_sep_mask('dataset/1blm/1blm.train.noise', 'dataset/1blm/1blm.train.gt',
    #                             'dataset/1blm/1blm.train.noise.sep_mask_all_sep', 'dataset/1blm/1blm.train.gt.sep_mask_all_sep',
    #                             sent_token='<sep>')
    # create_dataset_for_sep_mask('dataset/1blm/1blm.test.noise', 'dataset/1blm/1blm.test.gt',
    #                             'dataset/1blm/1blm.test.noise.sep_mask_all_sep', 'dataset/1blm/1blm.test.gt.sep_mask_all_sep',
    #                             sent_token='<sep>')

    # create_dataset_for_sep_mask('dataset/bea/bea2.noise', 'dataset/bea/bea2.gt',
    #                             'dataset/bea/bea2.noise.sep_mask', 'dataset/bea/bea2.gt.sep_mask')


if __name__ == '__main__':
    test()
