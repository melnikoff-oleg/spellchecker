from training.common_parts import char_based_model_init, get_sep_mask_training_dataset, launch_training
from model.spellcheck_model import CharBasedTransformerChecker
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    # data and model prep
    train_data, val_data = get_sep_mask_training_dataset(char_based=True)
    tokenizer, model = char_based_model_init(d_model=512)

    # set important learning params ------------------------------------------
    checkpoint = PATH_PREFIX + 'training/checkpoints/' + 'char-based-xl-explode_v1_9.pt'
    device_name = 'cuda:2'
    model_version = 2
    model_name = 'char-based-sep-mask'
    lr = 0.00005
    test_mode = True
    batch_size = 32
    num_epochs = 10
    print_n_batches = 2000
    st_epoch = 0
    spellcheck_class = CharBasedTransformerChecker
    save_model_interval = 30000

    # setup all remaining parts for learning
    launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint)


if __name__ == '__main__':
    main()
