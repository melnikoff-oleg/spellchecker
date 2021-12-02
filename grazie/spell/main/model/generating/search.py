from typing import Tuple

import torch


class Search:
    """
    Class for search algorithms
    Basically user needs to feed log_probs and perform a step several times
    Results can be found in hypotheses"""

    def __init__(self, vocab_size: int, search_size: int, repetition_penalty: float = 1.0):
        self._search_size = search_size
        self._vocab_size = vocab_size
        self._repetition_penalty = repetition_penalty or 1.0

    def step(self, log_probs: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take a single search step.

        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        raise NotImplementedError

    def _step_check(self, log_probs: torch.Tensor):
        assert log_probs.size() == (
            self.batch_size,
            self._vocab_size,
        ), f"log_probs must have shape {(self.batch_size, self._vocab_size)}, but {log_probs.size()} was given"

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        raise NotImplementedError

    @property
    def hypotheses(self) -> torch.Tensor:
        """Tensor of all tokens of the current hypotheses with shape (batch_size, seq_len) to make a batch for a model"""
        raise NotImplementedError

    @property
    def scores(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        raise NotImplementedError
