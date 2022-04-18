# use dataset with no tokenization breaks
# wrong token = wrong word
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=wJezVj1tYiBl
# тут все что нужно, датасет соберем из своих файлов, а дальше выравняем по лейблам токенизацию и гоу
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import torch
import nltk

# PATH_PREFIX = '/Users/olegmelnikov/PycharmProjects/spellcheker/grazie/spell/main/'
PATH_PREFIX = '/home/ubuntu/omelnikov/grazie/spell/main/'


def main():
    model_checkpoint = "distilbert-base-uncased"
    batch_size = 64


    datasets = load_dataset('json', data_files={'train': PATH_PREFIX + 'data/datasets/1blm/1blm.train.tagging',
                                               'val': PATH_PREFIX + 'data/datasets/bea/bea500.tagging',
                                               'test': PATH_PREFIX + 'data/datasets/bea/bea500.tagging'})

    print(datasets)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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

    print(tokenize_and_align_labels(datasets['train'][:5]))

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


    # to enable tensorboard should pass callback function to trainer

    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=2)

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-{'tagging'}",
        evaluation_strategy="steps",
        eval_steps=2000,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        push_to_hub=False,
    )

    from transformers import DataCollatorForTokenClassification

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # metric = load_metric("seqeval")
    example = datasets["train"][4]
    labels = [i for i in example["labels"]]

    def sample_metric(predictions=[[1, 2, 3], [2, 2]], references=[[1, 2, 3], [2, 2]]):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, j in zip(predictions, references):
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

    print(sample_metric(predictions=[labels], references=[labels]))

    import numpy as np

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

        results = sample_metric(predictions=true_predictions, references=true_labels)
        return results

    for i in range(0, 1):
        try:
            torch.cuda.set_device(i)
            model = model.to(torch.device(f'cuda:{i}'))
            print('DEVICE', model.device)

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
            break
        except Exception as e:
            print(e)

    trainer.evaluate()

    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
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

    results = sample_metric(predictions=true_predictions, references=true_labels)
    print(results)


    model.save_pretrained(PATH_PREFIX + 'training/checkpoints/bert-detector-0')

def infer_bert_tagger():
    # checkpoint = '/home/ubuntu/omelnikov/grazie/spell/main/training/checkpoints/bert-detector-0'
    checkpoint = '/home/ubuntu/omelnikov/distilbert-base-uncased-finetuned-tagging/checkpoint-124500'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=2)
    model = model.to(device)
    text = 'Secondly, I had to wait fourty - five minutes before the show finally began.'
    word_tokens = nltk.word_tokenize(text)
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_inputs = tokenizer(word_tokens, truncation=True, is_split_into_words=True, return_tensors='pt').to(device)["input_ids"]
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
    # print(tokenized_input.word_ids())
    # print(tokens)
    # print(torch.argmax(res.logits, dim=2))

if __name__ == '__main__':
    # main()
    infer_bert_tagger()
