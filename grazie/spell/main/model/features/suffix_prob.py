import copy
from math import log
from typing import List

import torch
from transformers import AutoModelForCausalLM

from grazie.spell.main.model.base import SpelledWord
from grazie.spell.main.model.features.base import BaseTransformerFeature
from grazie.spell.main.model.features.feature_tester import test_feature


class SuffixProbFeature(BaseTransformerFeature):
    def __init__(self, name: str = "distilgpt2"):
        super().__init__(name)
        self._model = AutoModelForCausalLM.from_pretrained(self._name)
        # self._tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def compute_candidates(self, text: str, spelled_word: SpelledWord, candidates: List[str]) -> List[float]:
        candidates_ids, masked_token_ids, left_idx = self.prepare(text, spelled_word, candidates)
        true_candidates_ids = copy.deepcopy(candidates_ids)
        # print('candidates_ids', candidates_ids)
        # print('masked_token_ids', masked_token_ids)
        # print('left_idx', left_idx)

        word_token_lens = [len(self._tokenizer(' ' + candidate)['input_ids']) for candidate in candidates]
        # print('word_token_lens', word_token_lens)
        candidates_ids = candidates_ids.convert_to_tensors("pt")
        # print('pt candidates_ids', candidates_ids)
        outputs = self._model(**candidates_ids)
        logits = outputs.logits

        log_probs = []
        probs = torch.softmax(logits, dim=2)
        # print(logits.shape, probs.shape)
        for cand_i, word_len in enumerate(word_token_lens):
            # print(self._tokenizer.convert_ids_to_tokens(true_candidates_ids["input_ids"][cand_i]))
            log_prob = []
            for i, token_id in enumerate(true_candidates_ids["input_ids"][cand_i]):
                if i < left_idx + word_len or token_id == self._tokenizer.pad_token_id:
                    continue
                # print(self._tokenizer.convert_ids_to_tokens(token_id))
                # print(probs[cand_i, i, :].shape, type(probs[cand_i, i, :]), probs[cand_i, i, :][0].item())
                # x = torch.argmax(probs[cand_i, i, :])
                # print(x.item())
                # print(self._tokenizer.convert_ids_to_tokens(torch.argmax(probs[cand_i, i, :]).item()))
                log_prob.append(log(probs[cand_i, i, token_id]))

            if log_prob:
                log_probs.append(sum(log_prob) / len(log_prob))
            else:
                log_probs.append(0.0)
            # print(candidates[cand_i])
        return log_probs


if __name__ == '__main__':
    test_feature(SuffixProbFeature())
