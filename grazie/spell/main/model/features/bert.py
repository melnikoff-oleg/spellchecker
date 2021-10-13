import copy
from math import log
from typing import List

import torch
from transformers import AutoModelForMaskedLM

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseTransformerFeature


class BertProbFeature(BaseTransformerFeature):
    def __init__(self, name: str = "bert-base-uncased"):
        super().__init__(name)
        self._model = AutoModelForMaskedLM.from_pretrained(self._name)

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        candidates_ids, masked_token_ids, left_idx = self.prepare(text, spelled_word, candidates)
        true_candidates_ids = copy.deepcopy(candidates_ids)

        word_token_lens = []
        for cand_i in range(len(candidates)):
            seq = candidates_ids["input_ids"][cand_i]
            pad_pos = seq.index(0) if 0 in seq else len(seq)
            # l = len(seq) - len(masked_token_ids) + 1 - (len(seq) - pad_pos)
            word_token_lens.append(pad_pos - len(masked_token_ids) + 1)

        idx = 0
        for cand_i, word_len in enumerate(word_token_lens):
            candidates_ids["input_ids"][idx][left_idx + word_len - 1] = masked_token_ids[left_idx]
            candidates_ids["attention_mask"][idx][left_idx + word_len - 1] = 0

            for i in range(1, word_len):
                ids = candidates_ids["input_ids"][idx].copy()
                mask = candidates_ids["attention_mask"][idx].copy()

                ids[left_idx + word_len - 1 - i] = masked_token_ids[left_idx]
                mask[left_idx + word_len - 1 - i] = 0

                candidates_ids["input_ids"].insert(idx, ids)
                candidates_ids["attention_mask"].insert(idx, mask)
                candidates_ids["token_type_ids"].insert(idx, candidates_ids["token_type_ids"][idx].copy())
            idx += word_len

        candidates_ids = candidates_ids.convert_to_tensors("pt")

        outputs = self._model(**candidates_ids)
        logits = outputs.logits

        log_probs: List[float] = []
        idx = 0
        for cand_i, word_len in enumerate(word_token_lens):
            log_prob = 0.0
            for token_i in range(word_len):
                token_logit = logits[idx][left_idx + token_i]
                token_probs = torch.softmax(token_logit, dim=0)
                log_prob += log(token_probs[true_candidates_ids["input_ids"][cand_i][left_idx + token_i]].item())
                idx += 1
            log_probs.append(log_prob)

        return log_probs
