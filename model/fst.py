from model.spellcheck_model import SpellCheckModelBase
from model.detector import BaseDetector, HunspellDetector
from model.candidator import BaseCandidator, HunspellCandidator
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch
from typing import List
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


class FastProdModel(SpellCheckModelBase):
    def __init__(self):
        self.detector: BaseDetector = HunspellDetector()
        self.candidator: BaseCandidator = HunspellCandidator()

        checkpoint_path = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-sent-distil-dec05_v0_81396.pt'
        config = BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
                                encoder_attention_heads=12, decoder_layers=3, decoder_ffn_dim=3072,
                                decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
                                activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
                                activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
                                use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                                is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2)
        model = BartForConditionalGeneration(config)
        model.load_state_dict(torch.load(checkpoint_path))
        self.device = torch.device('cuda')
        model = model.to(self.device)
        model.eval()
        self.ranker_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.ranker_model: BartForConditionalGeneration = model

    def correct(self, text: str, return_all_stages: bool = False) -> str:

        caps = (text.upper() == text)
        if caps:
            text = text.lower()

        spelled_words = self.detector.detect(text)
        candidates = self.candidator.get_candidates(text, spelled_words)

        _spelled_words, _candidates = [], []
        for idx, (spelled_word, cands) in enumerate(zip(spelled_words, candidates)):
            if len(candidates[idx]) > 0:
                _spelled_words.append(spelled_word)
                _candidates.append(cands)
        spelled_words, candidates = _spelled_words, _candidates

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

                input_texts = [spelled_word.word + ' </s> ' + text_pref + '<mask>' + text_suff for _ in cands]
                output_texts = [text_pref + syn + text_suff for syn in cands]

                texts += input_texts
                outs += output_texts
                cands_ranges += [(len(self.ranker_tokenizer.encode(text_pref[:-1])),
                                  len(self.ranker_tokenizer.encode(syn, add_special_tokens=False))) for syn in cands]
            else:
                print('Error with SpelledWord')
                print('SpelledWord:', spelled_word)
                print('Candidates:', cands)
                raise Exception

        batch_size = 16

        scores: List[List[float]] = [[] for _ in spelled_words]

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))

            encoded_input = self.ranker_tokenizer(texts[start: end], return_tensors='pt', truncation=True,
                                           padding=True).to(self.device)['input_ids']
            encoded_output = self.ranker_tokenizer(outs[start: end], return_tensors='pt', truncation=True,
                                            padding=True).to(self.device)['input_ids']

            # BART eval
            all_logits = self.ranker_model(encoded_input, labels=encoded_output).logits.cpu()

            for i, logits in enumerate(all_logits):
                ind = texts_inds[start + i]
                syn_range = cands_ranges[start + i]
                word_logits = logits[syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]
                log_probs = torch.log_softmax(word_logits, dim=1)
                word_log_prob = torch.tensor(0.0)
                for j, token_idx in enumerate(encoded_output[i][syn_range[0] - 1: syn_range[0] + syn_range[1] - 1]):
                    word_log_prob += log_probs[j, token_idx]
                scores[ind].append(word_log_prob.item())

        result: List[str] = ['' for _ in spelled_words]

        for i, cur_scores in enumerate(scores):
            mx = -1e18
            mx_ind = None
            for j, score in enumerate(cur_scores):
                print(candidates[i][j], score)
                if mx < score:
                    mx = score
                    mx_ind = j
            result[i] = candidates[i][mx_ind]

        corrections = result

        shift = 0
        res_text = text
        for i, spelled_word in enumerate(spelled_words):
            res_text = res_text[: shift + spelled_word.interval[0]] + corrections[i] + \
                       res_text[shift + spelled_word.interval[1]:]
            shift += len(corrections[i]) - len(spelled_word.word)

        if caps:
            res_text = res_text.upper()

        if return_all_stages:
            return res_text, spelled_words, candidates, corrections
        else:
            return res_text


def main():
    model: SpellCheckModelBase = FastProdModel()
    # text_noise = 'Moroever I have chosen this month because I think the weather will be fine.'
    # text_gt = 'Moreover I have chosen this month because I think the weather will be fine.'
    text_noise = 'I luk foward to receving from you'
    text_gt = 'I look forward to receiving from you'
    text_result = model.correct(text_noise)
    print(f'\nSpellcheck model testing\n\nModel: {str(model)}\n\n{text_noise} - Noised text\n{text_gt} - GT text'
          f'\n{text_result} - Result text')


if __name__ == '__main__':
    main()
