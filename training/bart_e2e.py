from model.spellcheck_model import BartChecker
from training.common_parts import bart_model_init, get_end_2_end_training_dataset, launch_training
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/spellchecker/'


def main():
    # data and model prep
    train_data, val_data = get_end_2_end_training_dataset()
    tokenizer, model = bart_model_init()

    # set important learning params ------------------------------------------
    checkpoint = PATH_PREFIX + 'training/checkpoints/' + 'bart-base_v1_4.pt'
    device_name = 'cuda:0'
    model_version = 2
    model_name = 'bart-base'
    lr = 0.00005
    test_mode = False
    st_epoch = 5
    batch_size = 32
    num_epochs = 10
    print_n_batches = 4000
    spellcheck_class = BartChecker
    save_model_interval = 30000

    # setup all remaining parts for learning
    launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint)


if __name__ == '__main__':
    main()
