from datasets.spell.main.training.trainer_transformer_seq2seq import train_model
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from datasets.spell.main.model.spellcheck_model import BART, SepMaskBART
from datasets.spell.main.training.data_processing import read_data
from transformers import RobertaTokenizer
import json
from transformers import BartConfig, BartForConditionalGeneration
import string
from datasets.spell.main.model.spellcheck_model import CharBasedTransformer

# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/grazie/spell/main/'
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


def main():
    train_data = read_data(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.gt.sep_mask',
                      noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.noise.sep_mask')
    val_data = read_data(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.gt.sep_mask',
                    noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.noise.sep_mask')

    create_vocab_files()
    tokenizer = BartTokenizer("url_vocab.json", "url_merges.txt")
    d_model = 512
    config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=d_model, encoder_layers=6, decoder_layers=6,
                        encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=d_model * 4,
                        decoder_ffn_dim=d_model * 4)
    model = BartForConditionalGeneration(config)

    # If needed take existing checkpoint
    checkpoint = PATH_PREFIX + 'training/checkpoints/char-based-xl-explode_v1_9.pt'
    model.load_state_dict(torch.load(checkpoint))
    print('Model loaded from', checkpoint)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_name = 'char-based-sep-mask'
    batch_size = 32
    num_epochs = 10
    st_epoch = 0
    print_n_batches = 2000
    num_sent = 1000000000
    model_version = 2
    save_model_interval = 30000
    test_mode = False
    train_data = train_data[:num_sent]
    num_batches_in_epoch = len(train_data) // batch_size

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00005)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_batches_in_epoch * 1,
                                                                   num_batches_in_epoch * num_epochs, num_epochs - 1)

    print(f'Start training. Num epocs: {num_epochs}, batch size: {batch_size}, num sents: {len(train_data)}')
    train_model(model, tokenizer, optimizer, scheduler, train_data, val_data, batch_size, print_n_batches, num_epochs,
                st_epoch, model_name, CharBasedTransformer, device, save_model=not test_mode, use_tensorboard=not test_mode,
                model_version=model_version, test_mode=test_mode, save_model_interval=save_model_interval)


if __name__ == '__main__':
    main()
