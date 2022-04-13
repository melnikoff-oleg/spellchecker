import random
from transformers import RobertaTokenizer
import string
import json
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import BartConfig, BartForConditionalGeneration
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import datetime

from datasets.spell.main.data.utils import get_texts_from_file
from datasets.spell.main.model.spellcheck_model import CharBasedTransformer
from datasets.spell.main.evaluation.evaluate import evaluate
from datasets.spell.main.training.data_processing import read_data_char_based
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


def train_model(model, tokenizer, train_data, val_data, num_epochs, batch_size, optimizer, scheduler,
                print_n_batches=2000, st_epoch=0, model_name='char_based_big', device=torch.device('cuda'),
                save_model=False, use_tensorboard=False, model_version=0):

    # Init tensorboard for logs writing
    if use_tensorboard:
        tb = torch.utils.tensorboard.SummaryWriter(log_dir=f'{PATH_PREFIX}training/tensorboard_logs/{model_name}/v{model_version}/st_epoch:{st_epoch}_date:{datetime.datetime.now().strftime("%m-%Y-%H-%M")}')

    num_batches = (len(train_data) + batch_size - 1) // batch_size
    for epoch in tqdm(range(st_epoch, st_epoch + num_epochs), desc='Epochs', leave=True):
        model.train()
        epoch_loss = 0
        for i in tqdm(range(num_batches), position=0, leave=True, desc='Batches'):
            batch = train_data[i * batch_size: min(i * batch_size + batch_size, len(train_data))]
            prefix = [i[0] for i in batch]
            suffix = [i[1] for i in batch]
            encoder_input = tokenizer(prefix, return_tensors='pt', padding=True).to(device)
            decoder_input = tokenizer(suffix, return_tensors='pt', padding=True).to(device)
            result = model(**encoder_input, labels=decoder_input['input_ids'])
            loss = result.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            epoch_loss += loss.cpu().item()
            batch_ind = num_batches * epoch + i

            # Printing all the stats and writing to tensorboard
            if batch_ind % print_n_batches == 0:
                print(f'\nTrain loss on batch {batch_ind}: {loss.cpu().item() / batch_size}')
                print(f'Learning rate: {scheduler.get_last_lr()[0]}')
                if use_tensorboard:
                    tb.add_scalar('Learning rate', scheduler.get_last_lr()[0], batch_ind)
                    tb.add_scalar('Train loss on batch', loss.cpu().item() / batch_size, batch_ind)

                # Calculate loss on validation data
                model.eval()
                with torch.no_grad():
                    val_batches = 10
                    batches = list(range(0, (len(val_data) + batch_size - 1) // batch_size))
                    random.shuffle(batches)
                    val_loss = 0
                    num_objects = 0
                    for j in batches[:val_batches]:
                        batch = val_data[j * batch_size: min(j * batch_size + batch_size, len(val_data))]
                        num_objects += len(batch)
                        prefix = [k[0] for k in batch]
                        suffix = [k[1] for k in batch]
                        encoder_input = tokenizer(prefix, return_tensors='pt', padding=True).to(device)
                        decoder_input = tokenizer(suffix, return_tensors='pt', padding=True).to(device)
                        result = model(**encoder_input, labels=decoder_input['input_ids'])

                        loss = result.loss
                        val_loss += loss.cpu().item()

                    val_loss /= num_objects

                    result_ids = model.generate(tokenizer([val_data[0][0], val_data[1][0], val_data[2][0]], return_tensors='pt', padding=True).to(device)["input_ids"],
                        num_beams=5, min_length=5, max_length=500)

                    ans = tokenizer.batch_decode(result_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ans_str = ''
                    for i in range(3):
                        ans_str += f'{val_data[i][0]} - Noised {i}\n{val_data[i][1]} - GT {i}\n{ans[i]} - Answer {i}'
                        if i < 2:
                            ans_str += '\n\n'

                    # Calculate metrics on test dataset
                    path_prefix = '/home/ubuntu/omelnikov/grazie/spell/main/'
                    char_based_transformer = CharBasedTransformer(model=model)
                    texts_gt, texts_noise = get_texts_from_file(path_prefix + 'data/datasets/bea/bea500.gt'), \
                                            get_texts_from_file(path_prefix + 'data/datasets/bea/bea500.noise')
                    evaluation_report = evaluate(char_based_transformer, texts_gt, texts_noise)
                    metrics = ['Precision', 'Recall', 'F_0_5', 'Word-level accuracy', 'Broken tokenization cases']
                    if use_tensorboard:
                        tb.add_text('Test sentence rewriting', ans_str, batch_ind)
                        tb.add_scalar("Val loss", val_loss, batch_ind)
                        for metric in metrics:
                            tb.add_scalar(metric, evaluation_report['Metrics'][metric], batch_ind)

                model.train()

        if save_model:
            model_path = f'{PATH_PREFIX}training/checkpoints/{model_name}_v{model_version}_{epoch}.pt'
            torch.save(model.state_dict(), model_path)
            print('Model saved in', model_path)
        else:
            print('Model was not saved')

        print('Train loss on epoch', epoch, ':', epoch_loss / len(train_data))
        tb.add_scalar("Train loss on epoch", epoch_loss / len(train_data), epoch)

    tb.close()


if __name__ == '__main__':
    train = read_data_char_based(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.gt', noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.train.noise')
    val = read_data_char_based(gt_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.gt', noise_path=PATH_PREFIX + 'data/datasets/1blm/1blm.test.noise')

    create_vocab_files()
    tokenizer = BartTokenizer("url_vocab.json", "url_merges.txt")
    config = BartConfig(vocab_size=tokenizer.vocab_size, d_model=256, encoder_layers=6, decoder_layers=6,
                        encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=1024, decoder_ffn_dim=1024)
    model = BartForConditionalGeneration(config)

    # If needed take existing checkpoint
    checkpoint = PATH_PREFIX + 'training/model_big_0_9.pt'
    model.load_state_dict(torch.load(checkpoint))
    print('Model loaded from', checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_name = 'char_based_big'
    batch_size = 64
    num_epochs = 10
    st_epoch = 10
    print_n_batches = 2000
    num_sent = 1000000000
    model_version = 0
    train = train[:num_sent]
    num_batches_in_epoch = len(train) // batch_size

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_batches_in_epoch, num_batches_in_epoch * num_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_batches_in_epoch * 3, num_batches_in_epoch * num_epochs)

    print(f'Start training. Num epocs: {num_epochs}, batch size: {batch_size}, num sents: {len(train)}')
    train_model(model, tokenizer, train, val, num_epochs, batch_size, optimizer, scheduler,
                print_n_batches=print_n_batches, st_epoch=st_epoch, model_name=model_name, device=device,
                save_model=True, use_tensorboard=True, model_version=model_version)
