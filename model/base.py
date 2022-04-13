from typing import Tuple
import attr


@attr.s(auto_attribs=True)
class SpelledWord:
    text: str
    interval: Tuple[int, int]
    word: str = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.word = self.text[self.interval[0]:self.interval[1]]
