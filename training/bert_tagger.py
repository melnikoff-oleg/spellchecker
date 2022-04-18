# use dataset with no tokenization breaks
# wrong token = wrong word
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=wJezVj1tYiBl
# тут все что нужно, датасет соберем из своих файлов, а дальше выравняем по лейблам токенизацию и гоу
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import torch
import nltk
import numpy as np
from typing import List
# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellchecker/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    tp, tn, fp, fn = 0, 0, 0, 0
    for i, j in zip(true_predictions, true_labels):
        for k, t in zip(i, j):
            if k == 1:
                if k == t:
                    tp += 1
                else:
                    fn += 1
            else:
                if k == t:
                    tn += 1
                else:
                    fp += 1
    return {
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def main():

    # model prep
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # data prep
    datasets = load_dataset('json', data_files={'train': PATH_PREFIX + 'datasets/1blm/1blm.train.tagging',
                                                'val': PATH_PREFIX + 'datasets/bea/bea500.tagging',
                                                'test': PATH_PREFIX + 'datasets/bea/bea500.tagging'})
    def tokenize_and_align_labels(examples):
        label_all_tokens = False
        try:
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        except Exception as e:
            print(examples["tokens"])
            raise e

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    # DEBUG
    # print(tokenize_and_align_labels(datasets['train'][:5]))
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # metrics calculation
    # DEBUG
    # example = datasets["train"][4]
    # labels = [i for i in example["labels"]]
    # print(sample_metric(predictions=[labels], references=[labels]))

    # set training params
    lr = 2e-5
    eval_steps = 2000
    batch_size = 64
    num_train_epochs = 2
    weight_decay = 0.01
    device = 3
    model_version = 2
    model_name = 'bert-detector'

    # launch learning
    torch.cuda.set_device(device)
    model = model.to(torch.device(f'cuda:{device}'))
    print(f'Device: {model.device}')
    # to enable custom tensorboard should pass callback function to trainer
    args = TrainingArguments(
        output_dir=f'{PATH_PREFIX}training/checkpoints/{model_name}_v{model_version}',
        evaluation_strategy="steps",
        save_strategy="epoch",
        logging_dir=f'{PATH_PREFIX}training/tensorboard_logs',
        eval_steps=eval_steps,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # eval model
    trainer.evaluate()
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    results = compute_metrics((predictions, labels))
    print(results)

    # save model
    model.save_pretrained(PATH_PREFIX + 'training/checkpoints/bert-detector-save2')


def infer_bert_tagger():

    # model prep
    # checkpoint = '/home/ubuntu/omelnikov/spellchecker/training/checkpoints/bert-detector-0'
    checkpoint = '/home/ubuntu/omelnikov/distilbert-base-uncased-finetuned-tagging/checkpoint-124500'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=2)
    model = model.to(device)
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # text prep
    text = 'Secondly, I had to wait fourty - five minutes before the show finally began.'
    word_tokens = nltk.word_tokenize(text)
    tokenized_inputs = tokenizer(word_tokens, truncation=True, is_split_into_words=True, return_tensors='pt').to(device)["input_ids"]

    # get model results
    res = model(tokenized_inputs)
    tokenized_input = tokenizer(word_tokens, is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    labels = torch.argmax(res.logits, dim=2).to('cpu').tolist()[0]
    # print(labels[0])
    word_ids = tokenized_input.word_ids()
    def spans(txt):
        tokens = nltk.word_tokenize(txt)
        offset = 0
        for token in tokens:
            offset = txt.find(token, offset)
            yield offset, offset + len(token)
            offset += len(token)
    word_spans = list(spans(text))
    final_labels = [0 for i in word_tokens]
    for ind in range(len(labels)):
        if labels[ind] == 1:
            final_labels[word_ids[ind]] = 1
    print(word_tokens)
    print(word_spans)
    print(final_labels)


if __name__ == '__main__':
    main()
