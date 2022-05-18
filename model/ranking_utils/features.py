import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from model.base import SpelledWord
from abc import ABC, abstractmethod
from typing import List
import nltk
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


class BaseFeature(ABC):
    @abstractmethod
    def compute_candidates(self, spelled_words: List[SpelledWord], candidates: List[List[str]]) -> List[List[float]]:
        raise NotImplementedError


class BartProbFeature(BaseFeature):
    def __init__(self, bart_type: str = 'std'):
        self.bart_type = bart_type
        if bart_type == 'std':
            checkpoint_path = 'facebook/bart-base'
            self.model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
            self.tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
        if bart_type == 'distilbart-de05':
            checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent-distil-dec05_v0_81396.pt'
            config = BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
                                encoder_attention_heads=12, decoder_layers=3, decoder_ffn_dim=3072,
                                decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
                                activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
                                activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
                                use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                                is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2)
            self.model = BartForConditionalGeneration(config)
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_candidates(self, spelled_words: List[SpelledWord], candidates: List[List[str]]) -> List[List[float]]:

        # prep data for BART
        texts = []
        outs = []
        texts_inds = []
        cands_ranges = []
        for i, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            text, start, finish = spelled_word.text, spelled_word.interval[0], spelled_word.interval[1]
            text_pref = text[: start]
            text_suff = text[finish:]
            if (start == 0 or text[start - 1] == ' ') and (finish == len(text) or not text[finish].isalpha()):
                texts_inds += [i for _ in cands]

                if self.bart_type == 'std':
                    input_texts = [text_pref + '<mask>' + text_suff for _ in cands]

                if self.bart_type == 'distilbart-de05':
                    input_texts = [spelled_word.word + ' </s> ' + text_pref + '<mask>' + text_suff for _ in cands]

                output_texts = [text_pref + syn + text_suff for syn in cands]

                texts += input_texts
                outs += output_texts
                cands_ranges += [(len(self.tokenizer.encode(text_pref[:-1])),
                                  len(self.tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]
            else:
                print('Error with SpelledWord')
                print('SpelledWord:', spelled_word)
                print('Candidates:', cands)
                raise Exception

        batch_size = 16

        scores: List[List[float]] = [[] for _ in spelled_words]

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))

            encoded_input = self.tokenizer(texts[start: end], return_tensors='pt', truncation=True,
                                           padding=True).to(self.device)['input_ids']
            encoded_output = self.tokenizer(outs[start: end], return_tensors='pt', truncation=True,
                                            padding=True).to(self.device)['input_ids']

            # BART eval
            all_logits = self.model(encoded_input, labels=encoded_output).logits.cpu()

            for i, logits in enumerate(all_logits):
                ind = texts_inds[start + i]
                syn_range = cands_ranges[start + i]
                word_logits = logits[syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]
                log_probs = torch.log_softmax(word_logits, dim=1)
                word_log_prob = torch.tensor(0.0)
                for j, token_idx in enumerate(encoded_output[i][syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]):
                    word_log_prob += log_probs[j, token_idx]
                # scores[ind].append(math.exp(word_log_prob.item()))
                scores[ind].append(word_log_prob.item())



        return scores


class LevenshteinFeature(BaseFeature):
    def compute_candidates(self, spelled_words: List[SpelledWord], candidates: List[List[str]]) -> List[List[float]]:

        scores: List[List[float]] = [[] for _ in spelled_words]
        for idx, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            for candidate in cands:
                dist = nltk.edit_distance(spelled_word.word, candidate, transpositions=True)
                scores[idx].append(dist)

        return scores


def test(feature: BaseFeature):
    print(f'Testing feature "{str(feature)}"')
    spelled_words: List[SpelledWord] = [SpelledWord(text='Hillo I am Charli', interval=(0, 5))]
    candidates: List[List[str]] = [['Hello', 'Hi', 'Hey', 'Harry']]
    scores = feature.compute_candidates(spelled_words, candidates)
    for spelled_word, cands, scors in zip(spelled_words, candidates, scores):
        print(f'Input text: {spelled_word.text}')
        print(f'Word with mistake: {spelled_word.word}')
        print(f'Candidates with scores:')
        for cand, score in zip(cands, scors):
            print(f'{cand} - {round(score, 2)}')


def main():
    test(LevenshteinFeature())


if __name__ == '__main__':
    main()
