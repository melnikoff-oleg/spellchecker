from typing import List

import syntok
import syntok.segmenter
from syntok.tokenizer import Tokenizer


class SyntokTextTokenizer:
    _tokenizer = Tokenizer(emit_hyphen_or_underscore_sep=True, replace_not_contraction=False)

    def split_to_sentences(self, text: str) -> List[str]:
        sentences: List[str] = []
        for par in syntok.segmenter.analyze(text):
            for sent in par:
                sent_tokens = list(sent)
                start, end = sent_tokens[0].offset, sent_tokens[-1].offset + len(sent_tokens[-1].value)
                sentences.append(text[start:end])
        return sentences

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for token in self._tokenizer.tokenize(text):
            tokens.append(token.value)
        return tokens
