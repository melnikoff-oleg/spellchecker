import random
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import datetime
import os

from data_utils.utils import get_texts_from_file
from model.spellcheck_model import BART
from evaluation.evaluate import evaluate

# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'

# one can make using tb through decorator


def train_model(model, tokenizer, optimizer, scheduler, train_data, val_data, batch_size: int = 32,
                print_n_batches: int = 2000, num_epochs: int = 10, st_epoch: int = 0, model_name: str = 'bart',
                spellcheck_class = BART, device=torch.device('cuda'), save_model=False, use_tensorboard=False,
                model_version=0, test_mode: bool = True, save_model_interval=None):

    # Init tensorboard for logs writing
    if use_tensorboard:
        tb = torch.utils.tensorboard.SummaryWriter(log_dir=f'{PATH_PREFIX}training/tensorboard_logs/{model_name}/v{model_version}/st_epoch:{st_epoch}_date:{datetime.datetime.now().strftime("%m-%Y-%H-%M")}')

    num_batches = (len(train_data) + batch_size - 1) // batch_size

    if save_model_interval is None:
        save_model_interval = num_batches - 1

    for epoch in tqdm(range(st_epoch, st_epoch + num_epochs), desc='Epochs', leave=True):
        model.train()
        epoch_loss = 0
        for i in tqdm(range(num_batches), position=0, leave=True, desc='Batches'):
            batch = train_data[i * batch_size: min(i * batch_size + batch_size, len(train_data))]
            prefix = [i[0] for i in batch]
            suffix = [i[1] for i in batch]
            # DEBUG
            # print('Training:', prefix[0], suffix[0], '\n')
            # sleep(5)
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

            # Save model
            if i % save_model_interval == 0:
                if save_model:
                    checkpoints_dir = f'{PATH_PREFIX}training/checkpoints/'
                    if not os.path.exists(checkpoints_dir):
                        os.makedirs(checkpoints_dir)
                    model_path = f'{checkpoints_dir}{model_name}_v{model_version}_{batch_ind}.pt'
                    torch.save(model.state_dict(), model_path)
                    print('Model saved in', model_path)
                else:
                    print('Model was not saved')

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
                    if test_mode:
                        val_batches = 1
                    else:
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

                    # DEBUG
                    # print('Check sent rewriting:', [val_data[0][0], val_data[1][0], val_data[2][0]], '\n')
                    # sleep(5)

                    result_ids = model.generate(tokenizer([val_data[0][0], val_data[1][0], val_data[2][0]], return_tensors='pt', padding=True).to(device)["input_ids"],
                        num_beams=5, min_length=5, max_length=500)

                    ans = tokenizer.batch_decode(result_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ans_str = ''
                    for i in range(3):
                        ans_str += f'{val_data[i][0]} - Noised {i}\n{val_data[i][1]} - GT {i}\n{ans[i]} - Answer {i}'
                        if i < 2:
                            ans_str += '\n\n'
                    if test_mode:
                        print(ans_str)
                        print('\n')

                    # Calculate metrics on test dataset
                    spellcheck_model = spellcheck_class(model=model, device=device)
                    if test_mode:
                        dataset_name = 'bea/bea2'
                        exp_save_dir = None
                    else:
                        dataset_name = 'bea/bea500'
                        exp_save_dir = PATH_PREFIX + f'data/experiments/{model_name}_v{model_version}_epoch_{epoch}/'
                    texts_gt, texts_noise = get_texts_from_file(PATH_PREFIX + f'data/datasets/{dataset_name}.gt'), \
                                            get_texts_from_file(PATH_PREFIX + f'data/datasets/{dataset_name}.noise')

                    evaluation_report = evaluate(spellcheck_model, texts_gt, texts_noise, exp_save_dir)

                    if test_mode:
                        print(evaluation_report)

                    metrics = ['Precision', 'Recall', 'F_0_5', 'Word-level accuracy', 'Broken tokenization cases']
                    if use_tensorboard:
                        tb.add_text('Test sentence rewriting', ans_str, batch_ind)
                        tb.add_scalar("Val loss", val_loss, batch_ind)
                        for metric in metrics:
                            tb.add_scalar(metric, evaluation_report['Metrics'][metric], batch_ind)

                model.train()

        print('Train loss on epoch', epoch, ':', epoch_loss / len(train_data))
        if use_tensorboard:
            tb.add_scalar("Train loss on epoch", epoch_loss / len(train_data), epoch)
    if use_tensorboard:
        tb.close()

