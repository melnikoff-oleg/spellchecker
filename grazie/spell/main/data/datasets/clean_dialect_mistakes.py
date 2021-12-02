import pandas as pd


def clean(gt_texts_path: str, noisy_texts_path: str, dialects_mapping_path: str):
    
    # adding dialects mappings
    br2am = {}
    am2br = {}
    df_dialects = pd.read_csv(dialects_mapping_path)
    for i, row in df_dialects.iterrows():
        br2am[row['british']] = row['american']
        am2br[row['american']] = row['british']

    with open(gt_texts_path) as f:
        gt_lines = f.readlines()
    with open(noisy_texts_path) as f:
        noise_lines = f.readlines()

    cleaned_noisy_path = noisy_texts_path + '_cleaned'
    with open(cleaned_noisy_path, 'w') as f:
        for gt, noise in zip(gt_lines, noise_lines):
            gt_words = gt.split()
            noise_words = noise.split()
            for ind in range(len(gt_words)):
                gt_word, noise_word = gt_words[ind].lower(), noise_words[ind].lower()
                if gt_word != noise_word:
                    if (br2am.get(gt_word) == noise_word) or (am2br.get(gt_word) == noise_word):
                        noise_words[ind] = gt_word
            f.write(' '.join(noise_words) + '\n')


if __name__ == '__main__':
    gt_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea4k'
    noisy_texts_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/datasets/test.bea4k.noise'
    dialects_mapping_path = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/data/british_american_spell/mapping.csv'
    clean(gt_texts_path, noisy_texts_path, dialects_mapping_path)
