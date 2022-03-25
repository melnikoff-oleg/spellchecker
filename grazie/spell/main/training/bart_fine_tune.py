from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import random
import string
import json
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import timeit
from transformers import get_linear_schedule_with_warmup


def train_model(model, tokenizer, train_data, val_data, num_epochs, batch_size, optimizer, scheduler, print_n_batches=2000,
                st_epoch=0, model_name='model_big_0', save_model=True):
    # model = model.to(device)

    tb = torch.utils.tensorboard.SummaryWriter()

    for epoch in tqdm(range(st_epoch, st_epoch + num_epochs), desc='Epochs', leave=True):
        model.train()
        epoch_loss = 0
        num_batches = (len(train_data) + batch_size - 1) // batch_size
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
            # printing all the stats and writing to tensorboard
            if batch_ind % print_n_batches == 0:
                print('\nLoss on batch', batch_ind, ':', loss.cpu().item() / batch_size)
                print('Learning rate:', scheduler.get_last_lr()[0])
                tb.add_scalar('Learning rate', scheduler.get_last_lr()[0], batch_ind)
                tb.add_scalar('Train loss on batch', loss.cpu().item() / batch_size, batch_ind)

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

                    tb.add_scalar("Val loss", val_loss, batch_ind)

                    result_ids = model.generate(tokenizer([val_data[0][0], val_data[1][0], val_data[2][0]], return_tensors='pt', padding=True).to(device)["input_ids"],
                        num_beams=5, min_length=5, max_length=100)

                    ans = tokenizer.batch_decode(result_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    ans_str = ''
                    for i in range(3):
                        ans_str += f'{val_data[i][0]} - Noised {i}\n{val_data[i][1]} - GT {i}\n{ans[i]} - Answer {i}'
                        if i < 2:
                            ans_str += '\n\n'

                    tb.add_text('Test sentence rewriting', ans_str, batch_ind)
                model.train()

        if save_model:
            model_path = f'{model_name}_{epoch}.pt'
            torch.save(model.state_dict(), model_path)
            print('Model saved in', model_path)
        else:
            print('Not saving model')

        print('Train loss on epoch', epoch, ':', epoch_loss / len(train_data))
        tb.add_scalar("Train loss on epoch", epoch_loss / len(train_data), epoch)

    tb.close()


def read_data(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i, j]))
    for ind, i in enumerate(data):
        data[ind] = (i[0].replace(' ', '_'), i[1].replace(' ', '_'))

    return data

if __name__ == '__main__':
    # path_prefix = '/Users/olegmelnikov/PycharmProjects/jb-spellchecker/'
    path_prefix = '/home/ubuntu/omelnikov/'
    # 'bea/bea60k.gt' 'bea/bea60k.noise' '1blm/1blm.train.gt' '1blm/1blm.train.noise' '1blm/1blm.test.gt' '1blm/1blm.test.noise'
    train = read_data(gt_path=path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.gt', noise_path=path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.noise')
    val = read_data(gt_path=path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.gt', noise_path=path_prefix + 'grazie/spell/main/data/datasets/bea/bea60k.noise')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    # dev_str = 'cuda:6'
    # print(dev_str)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # If needed take existing checkpoint
    # checkpoint = 'model_big_0_4.pt'
    # model.load_state_dict(torch.load(checkpoint))
    # print('Model loaded from', checkpoint)

    start = timeit.default_timer()

    model_name = 'model_bart_fine_tune_0'
    batch_size = 32
    num_epochs = 3
    st_epoch = 0
    print_n_batches = 10
    num_sent = 10000
    train = train[:num_sent]
    num_batches_in_epoch = len(train) // batch_size

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_batches_in_epoch, num_batches_in_epoch * num_epochs)

    print(f'Start training. Num epocs: {num_epochs}, batch size: {batch_size}, num sents: {len(train)}')
    train_model(model, tokenizer, train, val, num_epochs, batch_size, optimizer, scheduler, print_n_batches=print_n_batches, st_epoch=st_epoch, model_name=model_name, save_model=False)
    result_ids = model.generate(tokenizer([val[0][0]], return_tensors='pt').to(device)["input_ids"],
                                 num_beams=5, min_length=5, max_length=100)
    print('Quality check:')
    print('Query noise:', val[0][0][:-1])
    print('Query gt:', val[0][1][:-1])
    print('Answer:', tokenizer.batch_decode(result_ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False))


    stop = timeit.default_timer()

    print('Runtime:', stop - start)
