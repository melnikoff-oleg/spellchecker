from typing import List, Tuple, Dict

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class FillTextProbComputer:
    def __init__(self, checkpoint_path: str = "facebook/bart-base", device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path, add_prefix_space=True)
        self.model.eval()

    def create_prompt(self, text: str, start: int, end: int) -> str:
        return text[:start] + '<mask>' + text[end:]

    def log_probs(self, texts_with_ranges: List[Tuple[str, int, int]], candidates: List[List[str]]) -> List[List[float]]:
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        all_candidates = []
        for i, ((text, start, finish), cands) in enumerate(zip(texts_with_ranges, candidates)):
            text_start = text[:start]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                all_candidates += cands
                texts_inds += [i for _ in cands]

                input_text = self.create_prompt(text, start, finish)
                texts += [input_text for _ in cands]

                output_texts = [text_start + syn + text[finish:] for syn in cands]
                outs += output_texts

                cands_ranges += [(len(self.tokenizer.encode(text_start[:-1])) - 1, len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]

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

        scores: Dict[int, List[float]] = {}
        for i, logits in enumerate(all_logits):
            ind = texts_inds[i]
            if ind not in scores:
                scores[ind] = []

            syn_range = cands_ranges[i]
            word_logits = logits[syn_range[0] - 1:syn_range[0] + syn_range[1] - 1]
            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][syn_range[0]:syn_range[0] + syn_range[1]]):
                word_log_prob += log_probs[j, token_idx]

            scores[ind].append(word_log_prob.item())

        result: List[List[float]] = [[] for _ in texts_with_ranges]
        for i in scores:
            result[i] = scores[i]
        return result
