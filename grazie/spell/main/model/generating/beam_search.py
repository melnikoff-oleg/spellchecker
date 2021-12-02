from typing import Tuple

import torch
from torch import Tensor

from grazie.spell.main.model.generating.search import Search


class BeamSearch(Search):
    """Beam search algorithm with normalized by length scores"""

    def __init__(self, vocab_size: int, beam_size: int, repetition_penalty: float = 1.0):
        super().__init__(vocab_size, beam_size, repetition_penalty)

        self._initialied: bool = False
        self._scores: Tensor
        self._hypotheses: Tensor
        self._sort_mask: Tensor

    def _init_state(self, dtype: torch.dtype, device: torch.device):
        self._scores = torch.zeros(1, dtype=dtype, device=device)
        self._hypotheses = torch.empty(1, 0, dtype=torch.long, device=device)
        self._initialied = True

    def step(self, step_log_probs: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take a single search step.

        Args:
            step_log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step

            context: (batch_size, seq_len)
                initial context

        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        super()._step_check(step_log_probs)
        if not self._initialied:
            self._init_state(step_log_probs.dtype, step_log_probs.device)

        self.modify_score(step_log_probs, context)

        log_probs = step_log_probs + self._scores.unsqueeze(1)
        log_probs = log_probs.flatten()
        sample_scores, samples = torch.topk(
            log_probs,
            # Take more to ensure that we will keep search_size not terminated
            # min((1 + len(self._eos_ids)) * self._search_size, log_probs.size(0)),
            self._search_size,
            sorted=False,
        )
        sort_mask = torch.floor_divide(samples, self._vocab_size)
        samples.fmod_(self._vocab_size)

        self._init_sort_mask()
        self._update_state(samples, sample_scores, sort_mask)

        return self._sort_mask, samples

    def modify_score(self, scores: torch.Tensor, context: torch.Tensor = None):
        # repetition
        if self._repetition_penalty != 1.0:
            if context is not None:
                for i, con in enumerate(context):
                    prev_tokens = con.tolist()
                    prev_tokens_set = set(prev_tokens)
                    for pos, previous_token in enumerate(prev_tokens[::-1]):
                        if previous_token not in prev_tokens_set:
                            continue
                        prev_tokens_set.remove(previous_token)
                        # penalty = self._repetition_penalty
                        # this function penalty first token with self._repetition_penalty and decrease further with position increase
                        penalty = 1 / (pos + 1 / (self._repetition_penalty - 1)) + 1
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= penalty
                        else:
                            scores[i, previous_token].div(penalty)

            for i in range(self._hypotheses.shape[0]):
                for previous_token in set(self._hypotheses[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if scores[i, previous_token] < 0:
                        scores[i, previous_token] *= self._repetition_penalty
                    else:
                        # scores[i, previous_token] /= self._repetition_penalty
                        scores[i, previous_token].div(self._repetition_penalty)

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        assert (
            self._hypotheses is not None and self._hypotheses.size(1) > 0
        ), "Can't get last predictions if no steps have been performed"
        return self._hypotheses[:, -1]

    @property
    def hypotheses(self) -> torch.Tensor:
        return self._hypotheses

    @property
    def scores(self) -> torch.Tensor:
        return self._scores

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        if not self._initialied:
            return 1
        return self._scores.size(0)

    def _init_sort_mask(self):
        self._sort_mask = torch.arange(self.batch_size)

    def _update_state(self, samples: torch.Tensor, sample_scores: torch.Tensor, sort_mask: torch.Tensor):
        self._sort_state(sort_mask)

        self._scores = sample_scores
        self._hypotheses = torch.cat((self._hypotheses, samples.unsqueeze(1)), dim=1)

    def _sort_state(self, sort_mask: torch.Tensor = None):
        if sort_mask is None:
            _, sort_mask = torch.topk(self._scores, min(self._search_size, self._scores.size(0)))
        self._apply_slice_to_state(sort_mask)

    def _apply_slice_to_state(self, tensor_slice):
        self._scores = self._scores[tensor_slice]
        self._hypotheses = self._hypotheses[tensor_slice]
        if self._sort_mask is not None:
            self._sort_mask = self._sort_mask[tensor_slice]
