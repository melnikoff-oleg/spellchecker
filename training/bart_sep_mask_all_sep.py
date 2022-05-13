from model.spellcheck_model import BertBartChecker
from training.common_parts import bart_model_init, get_sep_mask_training_dataset, launch_training
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    special_tokens = ['<sep>']

    # data and model prep
    train_data, val_data = get_sep_mask_training_dataset(all_mistakes=True)
    tokenizer, model = bart_model_init()
    tokenizer.add_special_tokens(special_tokens)

    # set important learning params ------------------------------------------
    checkpoint = None
    device_name = 'cuda:1'
    model_version = 0
    model_name = 'bart-sep-mask-all-sep'
    lr = 0.0001
    test_mode = False
    batch_size = 32
    num_epochs = 10
    print_n_batches = 2000
    st_epoch = 0
    spellcheck_class = BertBartChecker
    save_model_interval = 30000

    # setup all remaining parts for learning
    launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint)


if __name__ == '__main__':
    main()
