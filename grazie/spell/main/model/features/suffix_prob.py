import copy
from math import log
from typing import List

import torch
from transformers import AutoModelForCausalLM

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseTransformerFeature


class SuffixProbFeature(BaseTransformerFeature):
    def __init__(self, name: str = "distilgpt2"):
        super().__init__(name)
        self._model = AutoModelForCausalLM.from_pretrained(self._name)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        candidates_ids, masked_token_ids, left_idx = self.prepare(text, spelled_word, candidates)
        true_candidates_ids = copy.deepcopy(candidates_ids)

        word_token_lens = [len(self._tokenizer(' ' + candidate)['input_ids']) for candidate in candidates]

        candidates_ids = candidates_ids.convert_to_tensors("pt")
        outputs = self._model(**candidates_ids)
        logits = outputs.logits

        log_probs = []
        probs = torch.softmax(logits, dim=2)
        for cand_i, word_len in enumerate(word_token_lens):
            log_prob = []
            for i, token_id in enumerate(true_candidates_ids["input_ids"][cand_i]):
                if i < left_idx + word_len or token_id == self._tokenizer.pad_token_id:
                    continue
                log_prob.append(log(probs[cand_i, i, token_id]))
            if log_prob:
                log_probs.append(sum(log_prob) / len(log_prob))
            else:
                log_probs.append(0.0)

        return log_probs
