from abc import ABC, abstractmethod
from model.base import SpelledWord
from typing import List, Dict
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from model.candidator import HunspellCandidator
from model.detector import HunspellDetector
import math


class Ranker(ABC):
    @abstractmethod
    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        raise NotImplementedError


class BartRanker(Ranker):
    def __init__(self, checkpoint_path: str = "facebook/bart-base", device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path, add_prefix_space=True)
        self.model.eval()

    def rank(self, text: str, spelled_words: List[SpelledWord], candidates: List[List[str]], **kwargs) -> List[str]:
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        all_candidates = []
        for i, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            text, start, finish = spelled_word.text, spelled_word.interval[0], spelled_word.interval[1]
            text_start = text[: start]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                all_candidates += cands
                texts_inds += [i for _ in cands]

                input_text = text[:start] + '<mask>' + text[finish:]
                texts += [input_text for _ in cands]

                output_texts = [text_start + syn + text[finish:] for syn in cands]
                outs += output_texts

                cands_ranges += [(len(self.tokenizer.encode(text_start[:-1])),
                                  len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]


        # DEBUG
        # print(f'Input BART texts: {texts}')

        encoded_input = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True, padding=True
        ).to(self.device)['input_ids']
        encoded_output = self.tokenizer.batch_encode_plus(
            outs,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True, padding=True
        ).to(self.device)['input_ids']

        all_logits = self.model(encoded_input, decoder_input_ids=encoded_output).logits.cpu()

        # DEBUG
        # print(f'Output BART logits: {all_logits}')

        scores: Dict[int, List[float]] = {}
        for i, logits in enumerate(all_logits):

            ind = texts_inds[i]
            if ind not in scores:
                scores[ind] = []

            # DEBUG
            # tokenized_input = self.tokenizer(outs[i])
            # tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
            # print(f'Output tokens:{tokens}')
            #
            # token_probs = []
            # all_probs = torch.softmax(logits, dim=1)
            # for j, token_idx in enumerate(encoded_output[i]):
            #     token_probs.append(all_probs[j, token_idx].item())
            #
            # print(f'Output tokens probs: {token_probs}')

            syn_range = cands_ranges[i]

            # DEBUG
            # print(f'Candidate tokens range: {syn_range}')

            word_logits = logits[syn_range[0] - 1:syn_range[0] + syn_range[1] - 1]
            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][syn_range[0]:syn_range[0] + syn_range[1]]):
                word_log_prob += log_probs[j, token_idx]
            # DEBUG
            # print(f'Candidate score: {word_log_prob.item()}')

            scores[ind].append(math.exp(word_log_prob.item()))

        result: List[str] = ['' for _ in spelled_words]
        for i in scores:
            mx = -1e18
            mx_ind = None
            for j, score in enumerate(scores[i]):
                # DEBUG
                # print(f'Candidate: {candidates[i][j]}, score: {score}')
                if mx < score:
                    mx = score
                    mx_ind = j
            result[i] = candidates[i][mx_ind]
        return result


def test():
    text = 'Harry warks in cofee shop'
    # spelled_words: List[SpelledWord] = [SpelledWord(text, (6, 11)), SpelledWord(text, (15, 20))]
    spelled_words: List[SpelledWord] = HunspellDetector().detect(text)
    detections = [i.word for i in spelled_words]
    candidates = HunspellCandidator().get_candidates(text, spelled_words)
    ranker = BartRanker(device=torch.device('cuda'))
    final_corrections = ranker.rank(text, spelled_words, candidates)
    print(f'Text: {text}\nHunspell detections: {detections}\nHunspell candidates: {candidates}\nBART ranker results: {final_corrections}\n')


if __name__ == '__main__':
    test()
