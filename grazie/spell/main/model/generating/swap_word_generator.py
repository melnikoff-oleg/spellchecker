import math
import os
from typing import Tuple, List

import torch
from transformers import BartTokenizer

from grazie.spell.main.model.generating.bart_model import BartGenerationModel
from grazie.spell.main.model.generating.completion_generation import CompletionGeneration
from grazie.spell.main.model.generating.info import GenerationInfo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SwapWordGenerator:
    def __init__(self, name: str, device: torch.device, num_beams: int = 5):
        self._device = device
        self.model = BartGenerationModel(name).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(name, add_prefix_space=True)
        self._beam_search = CompletionGeneration(self.model, self.tokenizer)
        self.num_beams = num_beams

    def generate(self, text: str, edges: Tuple[int, int]) -> List[Tuple[str, GenerationInfo]]:
        assert edges[0] < edges[1]
        if text.strip() == '':
            return []

        spelled_word = text[edges[0]: edges[1]]
        masked_text = text[:edges[0]] + " <mask>" + text[edges[1]:]
        # тут мы переводим маскированный текст в токены для BART-a
        encoder_ids = self.tokenizer.encode(masked_text, return_tensors="pt").to(self._device)

        # зачем здесь отбрасывается последний столбец матрицы? - хз
        # тут по идее мы ровно также закодировали только первую часть текста то что идет до токена маски
        decoder_ids = self.tokenizer.encode(text[:edges[0]], return_tensors="pt")[:, :-1].to(self._device)

        completions = []
        ans = []
        # здесь 3 - max tokens
        # что тут вызывается? - какой-то генератор… который елдит List[GenerationInfo]
        for not_terminated in self._beam_search.generate(encoder_ids, decoder_ids, self.num_beams, 3, spelled_word):
            # тут делаем проверку что в сгенеренном слове нет пробела тк иначе это уже 2 слова
            decoded_strings = [self.tokenizer.decode(info.ids)[1:] for info in not_terminated if ' ' not in self.tokenizer.decode(info.ids)[1:]]
            # decoded_strings = [self.tokenizer.decode(info.ids) for info in not_terminated]
            # completions += list(zip(decoded_strings, not_terminated))
            ans += decoded_strings

        # это тупо лист строк… каждая из которых была добавлена в на определенном числе токенов генерации
        return ans


def main():
    gen = SwapWordGenerator("facebook/bart-base", torch.device("cpu"), num_beams=10)
    # text = 'hello i am frim paris'
    # comps = gen.generate(text, (text.find(" frim"), text.find(" frim") + len(" frim")))


    # good
    # text = "This is a great opportunity for us to see the latest fashions and famous fashion models who we would like to have authograps from ."
    # word = ' authograps'

    # good
    # text = "I personally think the best would be to put the celebrities in cages and let people touch them , point and ask for authopraps ."
    # word = ' authopraps'

    # bad
    # text = "Yours sincerelly ,"
    # word = ' sincerelly'

    # bad
    # text = "Another succesfull carreer such as , film stars must also be balanced ."
    # word = ' succesfull'

    # bad
    text = "It is a coinsidence which we would be extremely happy to take advantage of ."
    word = ' coinsidence'




    comps = gen.generate(text, (text.find(word), text.find(word) + len(word)))
    # for comp in comps:
    #     print(comp[0], comp[1].ids, comp[1].probs(), comp[1].score(), math.exp(comp[1].score()))
    print(comps)


if __name__ == '__main__':
    main()
