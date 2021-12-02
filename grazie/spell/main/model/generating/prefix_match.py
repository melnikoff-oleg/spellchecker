from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, List, Set

from transformers import GPT2Tokenizer


class PrefixMatcher(ABC):
    def prefix_tokens(self, prefix: str, *args, **kwargs) -> List[int]:
        return self.prefix_tokens_by_err(prefix, err_limit=0)[1][0]

    def not_prefix_tokens(self, prefix: str, *args, **kwargs) -> List[int]:
        return self.prefix_tokens_by_err(prefix, err_limit=0)[0]

    @abstractmethod
    # @lru_cache(maxsize=50)
    def prefix_tokens_by_err(self, prefix: str, err_limit: int) -> Tuple[List[int], List[List[int]]]:
        raise NotImplementedError

    @staticmethod
    def err_cnt(s1: str, s2: str) -> int:
        cnt = 0
        for c1, c2 in zip(s1, s2):
            cnt += c1 != c2
        return cnt

    @staticmethod
    def levenshtein_dist(s1: str, s2: str) -> int:
        if len(s1) == 0 or len(s2) == 0:
            return 0

        matrix = [[0 for _ in range(len(s1) + 1)] for _ in range(len(s2) + 1)]
        prev_column = matrix[0]

        for i in range(len(s1)):
            prev_column[i + 1] = prev_column[i] + 1
        curr_column = matrix[1]

        for i2, c2 in enumerate(s2):
            curr_column[0] = prev_column[0] + 1

            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    curr_column[i1 + 1] = prev_column[i1]
                else:
                    change = 1 + prev_column[i1]
                    remove = 1 + prev_column[i1 + 1]
                    insert = 1 + curr_column[i1]

                    curr_column[i1 + 1] = min(change, remove, insert)

            if i2 != len(s2) - 1:
                prev_column = curr_column
                curr_column = matrix[i2 + 2]

        dist = min([row[-1] for row in matrix] + curr_column)
        return dist


# with bug in prefix_inds, use fuzzy match with zero errors
class StrictPrefixMatcher(PrefixMatcher):
    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokens = [tokenizer.decode(token_id) for token_id in range(tokenizer.vocab_size)]
        tokens_inds = sorted(enumerate(self.tokens), key=lambda p: p[1])
        self.orig_inds, self.tokens = map(list, zip(*tokens_inds))

        class Trie:
            def __init__(self):
                self.start = tokenizer.vocab_size
                self.finish = 0
                self.dict = {}

            def add(self, word, ind):
                self.start = min(self.start, ind)
                self.finish = max(self.finish, ind)
                if word == '':
                    return

                if word[0] not in self.dict:
                    self.dict[word[0]] = Trie()
                self.dict[word[0]].add(word[1:], ind)

            def prefix_inds(self, word):
                if word == '' or word[0] not in self.dict:
                    return self.start, self.finish
                return self.dict[word[0]].prefix_inds(word[1:])

        self.trie = Trie()
        for i, token in enumerate(self.tokens):
            self.trie.add(token, i)

    @lru_cache(maxsize=50)
    def prefix_tokens(self, prefix: str, *args, **kwargs):
        start, finish = self.trie.prefix_inds(prefix)
        return self.orig_inds[start:finish + 1]

    @lru_cache(maxsize=50)
    def not_prefix_tokens(self, prefix: str, *args, **kwargs):
        start, finish = self.trie.prefix_inds(prefix)
        return self.orig_inds[:start] + self.orig_inds[finish + 1:]
        # return sorted(self.orig_inds[:start] + self.orig_inds[finish + 1:])

    @lru_cache(maxsize=50)
    def prefix_tokens_by_err(self, prefix: str, err_limit: int) -> Tuple[List[int], List[List[int]]]:
        return self.not_prefix_tokens(prefix), [self.prefix_tokens(prefix)]


class FuzzyPrefixMatcher(PrefixMatcher):
    def __init__(self, tokenizer: GPT2Tokenizer, err_limit: int = 0, min_token_prefix_len: int = 3):
        self.tokenizer = tokenizer
        self.err_limit = err_limit
        self.min_token_prefix_len = min_token_prefix_len

        class Trie:
            def __init__(self):
                self.start = tokenizer.vocab_size
                self.finish = 0
                self.dict = {}
                self.terminated = False

            def add(self, word, ind):
                self.start = min(self.start, ind)
                self.finish = max(self.finish, ind)
                if word == '':
                    self.terminated = True
                    return

                if word[0] not in self.dict:
                    self.dict[word[0]] = Trie()
                self.dict[word[0]].add(word[1:], ind)

            def prefix_inds(self, word: str, err_limit: int = 0, depth: int = 0) -> List[Tuple[int, int, int]]:
                if word == '' or len(self.dict) == 0:
                    return [(self.start, self.finish + 1, 0)]

                if word[0] not in self.dict:
                    min_with_suffix = self.finish
                    for node in self.dict.values():
                        min_with_suffix = min(min_with_suffix, node.start)
                    return [(self.start, min_with_suffix, 0)]

                if word[0] == ' ':
                    return self.dict[word[0]].prefix_inds(word[1:], err_limit, depth + 1)

                result = []
                if err_limit > 0:
                    for symbol in self.dict:
                        if symbol == word[0]:
                            continue
                        if symbol.isalpha():
                            result += self.dict[symbol].prefix_inds(word[1:], err_limit - 1, depth + 1)  # replace
                            result += self.dict[symbol].prefix_inds(word, err_limit - 1, depth + 1)  # insert

                    result += self.prefix_inds(word[1:], err_limit - 1, depth + 1)  # delete

                    result = [(s, f, c + 1) for s, f, c in result]

                result += self.dict[word[0]].prefix_inds(word[1:], err_limit, depth + 1)  # correct

                # if self.terminated and depth >= min_token_prefix_len:
                #     result += [(self.start, self.start + 1, 0)]  # prefix of token

                return result

        self.tokens_by_id = [tokenizer.decode(token_id) for token_id in range(tokenizer.vocab_size)]
        tokens_inds = sorted(enumerate(self.tokens_by_id), key=lambda p: p[1])
        # self.orig_inds, self.tokens = map(list, zip(*tokens_inds))  # type: List[int], List[Any]
        self.orig_inds = [i for i, token in tokens_inds]
        str_sorted_tokens = [token for i, token in tokens_inds]

        self.trie = Trie()
        for i, token in enumerate(str_sorted_tokens):
            self.trie.add(token, i)

    @lru_cache(maxsize=50)
    def prefix_tokens_by_err(self, prefix: str, err_limit: int) -> Tuple[List[int], List[List[int]]]:
        if err_limit < 0:
            return self.orig_inds, []

        edges = self.trie.prefix_inds(prefix, err_limit)
        edges = sorted(edges)

        prev_start = 0
        result: List[List[int]] = [[] for _ in range(err_limit + 2)]

        for start, finish, err_count in edges:
            result[0] += self.orig_inds[prev_start:start]
            result[err_count + 1] += self.orig_inds[start:finish]
            prev_start = finish

        result[0] += self.orig_inds[prev_start:]

        not_matched = result[0]
        matched_by_err_count = []
        union_res: Set[int] = set()
        for r in result[1:]:
            err_res = set(r)
            matched_by_err_count.append(list(err_res - union_res))
            union_res |= err_res
        not_matched = list(set(not_matched) - union_res)

        return not_matched, matched_by_err_count
