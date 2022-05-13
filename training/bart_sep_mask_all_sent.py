from model.spellcheck_model import BertBartChecker
from transformers import BartConfig
from training.common_parts import bart_model_init, get_sep_mask_training_dataset, launch_training
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    # special_tokens = ['<sep>', '<sent>']
    # special_tokens_dict = {"sep_token": "<sep>", 'sent_tokчen': '<sent>'}
    # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>',
    #  'cls_token': '<s>', 'mask_token': '<mask>'}

    # data and model prep
    config = BartConfig(vocab_size = 50265, max_position_embeddings = 1024, encoder_layers = 6, encoder_ffn_dim = 3072,
                        encoder_attention_heads = 12, decoder_layers = 3, decoder_ffn_dim = 3072,
                        decoder_attention_heads = 12, encoder_layerdrop = 0.0, decoder_layerdrop = 0.0,
                        activation_function = 'gelu', d_model = 768, dropout = 0.1, attention_dropout = 0.0,
                        activation_dropout = 0.0, init_std = 0.02, classifier_dropout = 0.0, scale_embedding = False,
                        use_cache = True, num_labels=3, pad_token_id = 1, bos_token_id = 0, eos_token_id = 2,
                        is_encoder_decoder = True, decoder_start_token_id = 2, forced_eos_token_id = 2)
    checkpoint = PATH_PREFIX + 'training/checkpoints/bart-distil-dec05.pt'
    train_data, val_data = get_sep_mask_training_dataset(all_mistakes=True)
    tokenizer, model = bart_model_init(checkpoint, config)
    # tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.special_tokens_map)

    # set important learning params ------------------------------------------
    # checkpoint = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-214056-distil-dec-05.pt'
    device_name = 'cuda:0'
    model_version = 0
    model_name = 'bart-sep-mask-all-sent-distil-dec05'
    lr = 0.0001
    test_mode = False
    batch_size = 32
    num_epochs = 2
    print_n_batches = 2000
    st_epoch = 0
    spellcheck_class = BertBartChecker
    save_model_interval = 3876 # 4097007 # 3969782

    # setup all remaining parts for learning
    launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint)


if __name__ == '__main__':
    main()
