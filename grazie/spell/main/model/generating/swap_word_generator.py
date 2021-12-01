import os
from typing import Tuple, List

import torch
from transformers import BartTokenizer

from grazie.spell.main.model.generating.bart_model import BartGenerationModel
from grazie.spell.main.model.generating.completion_generation import CompletionGeneration
from grazie.spell.main.model.generating.info import GenerationInfo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SwapWordGenerator:
    def __init__(self, name: str, device: torch.device):
        self._device = device
        self.model = BartGenerationModel(name).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(name, add_prefix_space=True)
        self._beam_search = CompletionGeneration(self.model, self.tokenizer)

    def generate(self, text: str, edges: Tuple[int, int], num_beams: int = 10) -> List[Tuple[str, GenerationInfo]]:
        assert edges[0] < edges[1]
        if text.strip() == '':
            return []

        masked_text = text[:edges[0]] + " <mask>" + text[edges[1]:]
        encoder_ids = self.tokenizer.encode(masked_text, return_tensors="pt").to(self._device)
        decoder_ids = self.tokenizer.encode(text[:edges[0]], return_tensors="pt")[:, :-1].to(self._device)

        completions = []
        for not_terminated in self._beam_search.generate(encoder_ids, decoder_ids, num_beams, 3):
            decoded_strings = [self.tokenizer.decode(info.ids) for info in not_terminated]
            completions += list(zip(decoded_strings, not_terminated))

        return completions


def main():
    gen = SwapWordGenerator("facebook/bart-base", torch.device("cpu"))
    text = "hello my dear friend"
    comps = gen.generate(text, (text.find(" dear"), text.find(" dear") + len(" dear")))
    a = comps


if __name__ == '__main__':
    main()
