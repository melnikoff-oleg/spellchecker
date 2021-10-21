from typing import List

import nltk

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseFeature


class CountCandidatesLessEditDistFeature(BaseFeature):
    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        edit_dists = []
        for candidate in candidates:
            dist = nltk.edit_distance(spelled_word.word, candidate, transpositions=True)
            edit_dists.append(dist)

        less_edit_dist_count = []
        for candidate_ind, candidate in enumerate(candidates):
            cur_res = 0
            for other_candidate_ind, other_candidate in enumerate(candidates):
                if edit_dists[candidate_ind] > edit_dists[other_candidate_ind]:
                    cur_res += 1
            less_edit_dist_count.append(cur_res)

        return less_edit_dist_count
