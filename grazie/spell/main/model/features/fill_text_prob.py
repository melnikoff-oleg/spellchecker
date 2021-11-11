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

    def preprocess(self, texts_with_ranges: List[Tuple[str, int, int]],
                   candidates: List[List[str]]) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
        outs = []
        texts_inds = []
        cands_ranges = []
        for i, ((text, start, finish), cands) in enumerate(zip(texts_with_ranges, candidates)):
            text_start = text[:start]
            # remove if it is not separate phrase (space or start_of_text in the begin and end_of_text or not alpha after phrase)
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                texts_inds += [i for _ in cands]

                outs += [text_start + cand for cand in cands]

                start_tokens_len = len(self.tokenizer.encode(text_start[:-1])) - 1
                cands_ranges += [
                    (start_tokens_len, len(self.tokenizer.encode(cand, add_special_tokens=False))) for cand in cands
                ]
        return texts_inds, cands_ranges, outs

    def encode_prompt(self, texts):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            truncation=True, padding=True
        ).to(self.device)

    @staticmethod
    def candidate_scores(texts_inds: List[int], cands_ranges: List[Tuple[int, int]],
                         encoded_output: torch.Tensor, all_logits: torch.Tensor) -> Dict[int, List[float]]:
        scores: Dict[int, List[float]] = {i: [] for i in set(texts_inds)}
        for i, logits in enumerate(all_logits):
            ind = texts_inds[i]

            cand_range = cands_ranges[i]
            word_logits = logits[cand_range[0] - 1:cand_range[0] + cand_range[1] - 1]
            log_probs = torch.log_softmax(word_logits, dim=1)
            word_log_prob = torch.tensor(0.0)
            for j, token_idx in enumerate(encoded_output[i][cand_range[0]:cand_range[0] + cand_range[1]]):
                word_log_prob += log_probs[j, token_idx]

            scores[ind].append(word_log_prob.item())
        return scores

    def log_probs(self, texts_with_ranges: List[Tuple[str, int, int]], candidates: List[List[str]]) -> List[List[float]]:
        texts_inds, cands_ranges, outs = self.preprocess(texts_with_ranges, candidates)
        input_prompts = [self.create_prompt(*texts_with_ranges[i]) for i in texts_inds]

        encoded_input = self.encode_prompt(input_prompts)
        output_ids = self.encode_prompt(outs)['input_ids']

        all_logits = self.model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'],
                                decoder_input_ids=output_ids).logits.cpu()

        scores = self.candidate_scores(texts_inds, cands_ranges, output_ids, all_logits)
        return list(scores.values())


class SynFillTextProbComputer(FillTextProbComputer):
    def create_prompt(self, text: str, start: int, end: int) -> str:
        return text[start:end] + ' <sep> ' + text[:start] + '<mask>' + text[end:]
