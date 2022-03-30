from grazie.spell.main.training.trainer_transformer_seq2seq import train_model
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from grazie.spell.main.model.spellcheck_model import BART, SepMaskBART


# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'


def read_data(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i, j]))
    return data


def main():
    train_data = read_data(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.gt',
                      noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.noise.bart_sep_mask')
    val_data = read_data(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.gt',
                    noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.noise.bart_sep_mask')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    # If needed take existing checkpoint
    # checkpoint = PATH_PREFIX + 'training/model_big_0_9.pt'
    # model.load_state_dict(torch.load(checkpoint))
    # print('Model loaded from', checkpoint)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_name = 'bart-sep-mask'
    batch_size = 64
    num_epochs = 5
    st_epoch = 0
    print_n_batches = 2000
    num_sent = 1000000000
    model_version = 0
    train_data = train_data[:num_sent]
    num_batches_in_epoch = len(train_data) // batch_size

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_batches_in_epoch * 1, num_batches_in_epoch * num_epochs)

    print(f'Start training. Num epocs: {num_epochs}, batch size: {batch_size}, num sents: {len(train_data)}')
    train_model(model, tokenizer, optimizer, scheduler, train_data, val_data, batch_size, print_n_batches, num_epochs,
                st_epoch, model_name, SepMaskBART, device, save_model=True, use_tensorboard=True,
                model_version=model_version, test_mode=False)


if __name__ == '__main__':
    main()
