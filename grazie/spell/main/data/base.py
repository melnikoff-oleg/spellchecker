from typing import List, Tuple

import attr


@attr.s(auto_attribs=True, frozen=True)
class Spell:
    spelled: str
    correct: str


@attr.s(auto_attribs=True, frozen=True)
class SpelledText:
    text: str
    spells: List[Spell]
