from typing import List, Optional, Union

import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from grazie.spell.main.model.generating.model import GenerationModel


class GPT2GenerationModel(GenerationModel):
    class GPT2State(GenerationModel.GenerationState):
        def __init__(self):
            self.past: Optional[List[List[torch.Tensor]]] = None

        def update(self, sort_mask: Union[List[int], torch.Tensor]):
            if self.past is not None:
                self.past = [[m[sort_mask].contiguous() for m in mem] for mem in self.past]

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def to(self, device: torch.device) -> 'GPT2GenerationModel':
        self.model.to(device)
        return self

    def create_state(self) -> GPT2State:
        return self.GPT2State()

    def all_probs(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.model(input_ids).logits
        scores = F.softmax(logits, dim=2)
        return scores

    def next_probs(self, input_ids: torch.Tensor, state: GenerationModel.GenerationState, **kwargs) -> torch.Tensor:
        assert isinstance(state, self.GPT2State)

        if state.past is not None:
            inputs = input_ids[:, state.past[0][0].shape[-2]:]
            model_out = self.model(inputs, state.past)
        else:
            model_out = self.model(input_ids)
        logits, state.past = model_out.logits, model_out.past_key_values
        last_logits = logits[:, -1, :]
        scores = F.softmax(last_logits, dim=1)
        return scores
