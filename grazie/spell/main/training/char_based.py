from grazie.spell.main.training.trainer_transformer_seq2seq import train_model
from transformers import RobertaTokenizer
import string
import json
from grazie.spell.main.model.spellcheck_model import CharBasedTransformer
import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup



# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'

# Char-based tokenizer
class BartTokenizer(RobertaTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# Vocabs for char-based tokenizer
def create_vocab_files():
    chars = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + list(string.punctuation) + list(string.digits) + list(string.ascii_lowercase) + list(string.ascii_uppercase)
    url_vocab = {c: i for i, c in enumerate(chars)}
    with open("url_vocab.json", 'w') as json_file:
      json.dump(url_vocab, json_file)

    merges = "#version: 0.2\n"
    with open("url_merges.txt", 'w') as f:
        f.write(merges)

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
                      noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.noise')
    val_data = read_data(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.gt',
                    noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.noise')

    create_vocab_files()
    tokenizer = BartTokenizer("url_vocab.json", "url_merges.txt")
    d_model = 128
    config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=d_model, encoder_layers=6, decoder_layers=6,
                        encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=d_model * 4,
                        decoder_ffn_dim=d_model * 4)
    model = BartForConditionalGeneration(config)


    # If needed take existing checkpoint
    # checkpoint = PATH_PREFIX + 'training/model_big_0_9.pt'
    # model.load_state_dict(torch.load(checkpoint))
    # print('Model loaded from', checkpoint)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print('Device:', device)

    model_name = 'char-based-med'
    batch_size = 64
    num_epochs = 3
    st_epoch = 0
    print_n_batches = 10
    num_sent = 1000
    model_version = 0
    train_data = train_data[:num_sent]
    num_batches_in_epoch = len(train_data) // batch_size

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_batches_in_epoch * 1, num_batches_in_epoch * num_epochs)

    print(f'Start training. Model name: {model_name}, Num epocs: {num_epochs}, batch size: {batch_size}, num sents: '
          f'{len(train_data)}')
    train_model(model, tokenizer, optimizer, scheduler, train_data, val_data, batch_size, print_n_batches, num_epochs,
                st_epoch, model_name, CharBasedTransformer, device, save_model=False, use_tensorboard=False,
                model_version=model_version, test_mode=True)


if __name__ == '__main__':
    main()
