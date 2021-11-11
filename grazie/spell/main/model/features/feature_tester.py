from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


def test_feature(feature: BaseFeature):
    texts = ['dear students , i im your teacher', 'I do to school everyday', 'america plains to attack iran']
    spelled_words = [SpelledWord(texts[0], (18, 20)), SpelledWord(texts[1], (2, 4)), SpelledWord(texts[2], (8, 14))]
    candidates = [['am', 'was', 'go', 'will'], ['go', 'done', 'dot'], ['plans', 'paints', 'plants']]
    passed_test_cases = True
    for text, spelled_word, cands in zip(texts, spelled_words, candidates):
        res = feature.compute_candidates(text, spelled_word, cands)

        for i in range(1, len(cands)):
            diff = abs(res[0] - res[i])
            if res[0] < res[i] or diff < abs(res[i]) / 4:
                print(f'Bad test case founded!\nText: {text}, Incorrect word: {spelled_word.word}\nCandidates: {cands}\nTheir scores: {res}')
                passed_test_cases = False
                break

    if passed_test_cases:
        print('All test cases passed successfully!')
