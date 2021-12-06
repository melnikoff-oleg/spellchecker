from typing import List

import attr


@attr.s(auto_attribs=True, frozen=True)
class Spell:
    spelled: str
    correct: str
    start: int


@attr.s(auto_attribs=True, frozen=True)
class SpelledText:
    text: str
    spells: List[Spell]
