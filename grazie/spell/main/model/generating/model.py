from abc import ABC, abstractmethod
from typing import List, Union

import torch


class GenerationModel(ABC):
    class GenerationState:
        def update(self, sort_mask: Union[List[int], torch.Tensor]):
            pass

    def to(self, device: torch.device) -> 'GenerationModel':
        return self

    def create_state(self) -> GenerationState:
        return self.GenerationState()

    @abstractmethod
    def all_probs(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def next_probs(self, data: torch.Tensor, state: GenerationState, **kwargs) -> torch.Tensor:
        raise NotImplementedError