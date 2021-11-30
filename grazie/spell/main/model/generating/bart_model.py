from typing import List, Optional, Union

import torch
from torch.nn import functional as F
from transformers import BartForConditionalGeneration

from grazie.spell.main.model.generating.model import GenerationModel


class BartGenerationModel(GenerationModel):
    class BartState(GenerationModel.GenerationState):
        def __init__(self):
            self.past: Optional[List[List[torch.Tensor]]] = None

        def update(self, sort_mask: Union[List[int], torch.Tensor]):
            if self.past is not None:
                self.past = [[m[sort_mask].contiguous() for m in mem] for mem in self.past]

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

    def to(self, device: torch.device) -> 'BartGenerationModel':
        self.model.to(device)
        return self

    def create_state(self) -> BartState:
        return self.BartState()

    def all_probs(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.model(input_ids).logits
        scores = F.softmax(logits, dim=2)
        return scores

    def next_probs(self, input_ids: torch.Tensor, state: GenerationModel.GenerationState,
                   encoder_input_ids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert encoder_input_ids is not None
        assert isinstance(state, self.BartState)

        if state.past is not None:
            inputs = input_ids[:, state.past[0][0].shape[-2]:]
            model_out = self.model(
                encoder_input_ids, decoder_input_ids=inputs, past_key_values=state.past, use_cache=True
            )
        else:
            model_out = self.model(input_ids)
        logits, state.past = model_out.logits, model_out.past_key_values
        last_logits = logits[:, -1, :]
        scores = F.softmax(last_logits, dim=1)
        return scores
