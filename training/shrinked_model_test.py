from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch
import time
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    configs = [BartConfig(vocab_size = 50265, max_position_embeddings = 1024, encoder_layers = 6, encoder_ffn_dim = 3072,
                        encoder_attention_heads = 12, decoder_layers = 3, decoder_ffn_dim = 3072,
                        decoder_attention_heads = 12, encoder_layerdrop = 0.0, decoder_layerdrop = 0.0,
                        activation_function = 'gelu', d_model = 768, dropout = 0.1, attention_dropout = 0.0,
                        activation_dropout = 0.0, init_std = 0.02, classifier_dropout = 0.0, scale_embedding = False,
                        use_cache = True, num_labels=3, pad_token_id = 1, bos_token_id = 0, eos_token_id = 2,
                        is_encoder_decoder = True, decoder_start_token_id = 2, forced_eos_token_id = 2),
               BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
                          encoder_attention_heads=12, decoder_layers=5, decoder_ffn_dim=3072,
                          decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
                          activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
                          activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
                          use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                          is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2),
               BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072,
                          encoder_attention_heads=12, decoder_layers=6, decoder_ffn_dim=3072,
                          decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
                          activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.0,
                          activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False,
                          use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                          is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2),
               ]
    checkpoints = [
        'training/checkpoints/bart-distil-dec05.pt',
        'training/checkpoints/bart-distil-dec-085-test.pt',
        'training/checkpoints/bart-distil-dec-full-test.pt',
    ]

    for config, checkpoint in zip(configs, checkpoints):
        checkpoint = PATH_PREFIX + checkpoint

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartForConditionalGeneration(config)
        model.load_state_dict(torch.load(checkpoint))

        device = torch.device('cuda')
        model.to(device)

        text = 'Hello <mask> friend, we go to lake'

        start_time = time.time()
        ans_ids = model.generate(tokenizer([text], return_tensors='pt').to(device)["input_ids"], num_beams=5, min_length=5,
                                 max_length=500)
        ans_tokens = tokenizer.batch_decode(ans_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("--- %s seconds ---" % (time.time() - start_time))
        res = ' '.join(ans_tokens)
        print(text)
        print(res)


if __name__ == '__main__':
    main()
