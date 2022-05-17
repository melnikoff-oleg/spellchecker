from model.spellcheck_model import BertBartChecker
from training.common_parts import bart_model_init, get_sep_mask_training_dataset, launch_training
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    # BART special tokens
    # special_tokens_dict = {"sep_token": "<sep>", 'sent_tokчen': '<sent>'}
    # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>',
    #  'cls_token': '<s>', 'mask_token': '<mask>'}

    # data and model prep
    train_data, val_data = get_sep_mask_training_dataset(all_mistakes=True, sent=True)
    tokenizer, model = bart_model_init()
    tokenizer.add_tokens(['<sent>'])
    model.resize_token_embeddings(len(tokenizer))

    # set important learning params ------------------------------------------
    # checkpoint = PATH_PREFIX + 'training/checkpoints/bart-sep-mask-all-214056-distil-dec-05.pt'
    device_name = 'cuda:0'
    model_version = 0
    model_name = 'bart-sep-mask-all-2'
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
                    save_model_interval, lr, checkpoint=None)


if __name__ == '__main__':
    main()
