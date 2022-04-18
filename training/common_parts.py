from model.spellcheck_model import CharBasedTransformerChecker
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from data_utils.utils import get_parallel_texts_from_files
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import torch
from training.trainer_transformer_seq2seq import train_model
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellcheker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'


def char_based_model_init(d_model=256):
    tokenizer = CharBasedTransformerChecker.BartTokenizer(
        PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_vocab.json",
        PATH_PREFIX + "data_utils/char_based_transformer_vocab/url_merges.txt"
    )
    config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=d_model, encoder_layers=6, decoder_layers=6,
                        encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=d_model * 4,
                        decoder_ffn_dim=d_model * 4)
    model = BartForConditionalGeneration(config)
    return tokenizer, model


def bart_model_init():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    return tokenizer, model


def get_end_2_end_training_dataset(char_based: bool = False):
    train = get_parallel_texts_from_files(file1_path=PATH_PREFIX + 'datasets/1blm/1blm.train.gt',
                                          file2_path=PATH_PREFIX + 'datasets/1blm/1blm.train.noise',
                                          char_based=char_based)
    val = get_parallel_texts_from_files(file1_path=PATH_PREFIX + 'datasets/1blm/1blm.test.gt',
                                        file2_path=PATH_PREFIX + 'datasets/1blm/1blm.test.noise',
                                        char_based=char_based)
    return train, val


def get_sep_mask_training_dataset(char_based: bool = False):
    train = get_parallel_texts_from_files(file1_path=PATH_PREFIX + 'datasets/1blm/1blm.train.gt.sep_mask.shuffled',
                                          file2_path=PATH_PREFIX + 'datasets/1blm/1blm.train.noise.sep_mask.shuffled',
                                          char_based=char_based)
    val = get_parallel_texts_from_files(file1_path=PATH_PREFIX + 'datasets/1blm/1blm.test.gt.sep_mask.shuffled',
                                        file2_path=PATH_PREFIX + 'datasets/1blm/1blm.test.noise.sep_mask.shuffled',
                                        char_based=char_based)
    return train, val


def launch_training(model, tokenizer, train_data, val_data, batch_size, print_n_batches,
                    num_epochs, st_epoch, model_name, spellcheck_class, device_name, test_mode, model_version,
                    save_model_interval, lr, checkpoint):

    num_sent = 1000 if test_mode else 1000000000
    train_data = train_data[:num_sent]
    num_batches_in_epoch = len(train_data) // batch_size
    num_warmup_steps = num_batches_in_epoch * 1
    num_training_steps = num_batches_in_epoch * num_epochs
    num_cycles = num_epochs - 1
    last_epoch = -1
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                   num_warmup_steps=num_warmup_steps,
                                                                   num_training_steps=num_training_steps,
                                                                   num_cycles=num_cycles, last_epoch=last_epoch)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        print('Model loaded from', checkpoint)
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'Start training. Num epochs: {num_epochs}, batch size: {batch_size}, num sentences: {len(train_data)}')
    train_model(model=model, tokenizer=tokenizer, optimizer=optimizer, scheduler=scheduler, train_data=train_data,
                val_data=val_data, batch_size=batch_size, print_n_batches=print_n_batches, num_epochs=num_epochs,
                st_epoch=st_epoch, model_name=model_name, spellcheck_class=spellcheck_class,
                device=device, save_model=not test_mode, use_tensorboard=not test_mode, model_version=model_version,
                test_mode=test_mode, save_model_interval=save_model_interval)
